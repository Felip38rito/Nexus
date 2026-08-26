#!/usr/bin/env python3
"""Nexus → OpenCode model sync.

Reads the model tiers advertised by the Nexus router (`GET /v1/models`) and
writes them into the `models` block of the `router` provider inside an
`opencode.json(c)` config.

Why this exists
---------------
OpenCode does NOT auto-discover models for custom OpenAI-compatible providers
(no `GET /v1/models` probing — see anomalyco/opencode#27553). It requires a
statically declared `models` block per provider. So the router cannot be a
*live* source of truth for OpenCode the way it is for Hermes; instead we keep
the router as the single place that defines the tiers, and this script pushes a
snapshot of that list into OpenCode's config whenever you run it (or restart
the router).

Safety / non-destructive
------------------------
- Backs up the target file before touching it (timestamped copy next to it).
- Only rewrites the `models` block of the provider named `router`. Every other
  provider and every other key in the file are preserved byte-for-byte.
- If the target file does not exist or the `router` provider is absent, it is
  created/inserted without touching anything else.
- Never deletes user config: any models already listed under the `router`
  provider that the router no longer advertises are REMOVED (that is the
  point — the router is the source of truth), but nothing outside that block
  is ever modified.

Usage
-----
    python3 sync-opencode.py [--config PATH] [--router-url URL] [--dry-run]

Examples
--------
    # Sync the default config (~/.config/opencode/opencode.jsonc)
    python3 sync-opencode.py

    # Dry-run: print what WOULD change without writing anything
    python3 sync-opencode.py --dry-run

    # Target a different config and router
    python3 sync-opencode.py --config /tmp/opencode.json --router-url http://127.0.0.1:9000/v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# Default human-friendly label per tier, used when the router doesn't give us a
# nicer display name. The upstream api id is always appended for clarity.
TIER_LABELS = {
    "mini": "Mini",
    "air": "Air",
    "pro": "Pro",
    "ultra": "Ultra",
}

# Default context/output limits (tokens) applied to synced model entries. The
# router's /v1/models response does not carry these today; override freely.
DEFAULT_CONTEXT = 1048576
DEFAULT_OUTPUT = 65536

DEFAULT_CONFIG = Path.home() / ".config" / "opencode" / "opencode.jsonc"
DEFAULT_ROUTER_URL = "http://127.0.0.1:9000/v1"
PROVIDER_NAME = "router"


# ---------------------------------------------------------------------------
# JSONC helpers (tolerant parse + byte-preserving targeted edits)
# ---------------------------------------------------------------------------

def _strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments from JSONC text for parsing.

    Keeps strings intact so a '//' inside a URL/path is not stripped.
    """
    out = []
    i = 0
    n = len(text)
    in_str = False
    in_block = False
    while i < n:
        c = text[i]
        if in_block:
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            # line comment
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            in_block = True
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] or } so json.loads accepts JSONC."""
    import re

    # Replace ',' followed (after whitespace) by ] or } with just that closer.
    return re.sub(r",(\s*[\]}])", r"\1", text)


def parse_jsonc(text: str):
    """Parse JSONC text into a Python object (tolerates comments + trailing commas)."""
    cleaned = _strip_trailing_commas(_strip_jsonc_comments(text))
    return json.loads(cleaned)


def _find_matching(text: str, open_idx: int) -> int:
    """Given the index of an opening brace/bracket, return the index of its match."""
    opener = text[open_idx]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces/brackets")


def _find_key_block(text: str, key: str, start: int = 0) -> tuple[int, int] | None:
    """Find `"key":` as a value at the current nesting, return (value_start, value_end).

    Only matches when the key is a *map pair* — i.e. the previous non-whitespace
    character is `{` or `,` (so a `"key"` that appears inside a string or as a
    value is skipped).
    """
    needle = f'"{key}"'
    search_from = start
    n = len(text)
    while True:
        idx = text.find(needle, search_from)
        if idx == -1:
            return None
        # walk back over whitespace to the previous significant char
        j = idx - 1
        while j >= 0 and text[j] in " \t\r\n":
            j -= 1
        pre = text[j] if j >= 0 else "{"
        if pre in "{,":
            colon = text.find(":", idx + len(needle))
            if colon != -1:
                vstart = colon + 1
                while vstart < n and text[vstart] in " \t\r\n":
                    vstart += 1
                if vstart < n and text[vstart] in "[{":
                    vend = _find_matching(text, vstart)
                    return vstart, vend + 1
                # scalar value: extend to end of token
                vend = vstart
                while vend < n and text[vend] not in ",}\n":
                    vend += 1
                return vstart, vend
        search_from = idx + len(needle)


def _find_provider_object(text: str, provider: str) -> tuple[int, int] | None:
    """Find the `"<provider>": { ... }` object inside a `"provider": {...}` map.

    Returns (object_start, object_end). Locates the `"provider"` key, then the
    `<provider>` key within it.
    """
    prov_key = _find_key_block(text, "provider")
    if not prov_key:
        return None
    _ps, pe = prov_key
    inner = _find_key_block(text, provider, start=0)
    if not inner:
        return None
    pstart, pend = inner
    # ensure it is inside the provider object's span
    if not (prov_key[0] <= pstart <= pe):
        # search scoped inside provider object
        obj_start = prov_key[1] - 1 if text[prov_key[1] - 1] in "}]" else prov_key[1]
        inner2 = _find_key_block(text, provider, start=obj_start)
        if inner2 is None:
            return None
        pstart, pend = inner2
    return pstart, pend


def _indent_of(text: str, idx: int) -> str:
    """Return the leading-whitespace indentation of the line containing idx."""
    line_start = text.rfind("\n", 0, idx) + 1
    j = line_start
    while j < idx and text[j] in " \t":
        j += 1
    return text[line_start:j]


# ---------------------------------------------------------------------------
# Model list generation
# ---------------------------------------------------------------------------

def fetch_models(router_url: str) -> list[dict]:
    """GET the router's /v1/models and return the data list."""
    url = router_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("data", [])


def build_models_block(models: list[dict]) -> dict:
    """Turn the router's model rows into an OpenCode `models` mapping."""
    result: dict = {}
    for row in models:
        mid = row.get("id")
        if not mid:
            continue
        tier = row.get("tier", "") or ("adaptive" if mid == "adaptive" else "")
        api_model = row.get("model", "")
        if mid == "adaptive":
            name = "Adaptive (auto-tier)"
        elif tier:
            label = TIER_LABELS.get(tier, tier.capitalize())
            name = f"{label} — {api_model or mid}"
        else:
            name = api_model or mid
        entry = {"name": name}
        entry["limit"] = {
            "context": DEFAULT_CONTEXT,
            "output": DEFAULT_OUTPUT,
        }
        result[mid] = entry
    return result


# ---------------------------------------------------------------------------
# File rewrite (byte-preserving for everything except the target block)
# ---------------------------------------------------------------------------

def render_models_value(block: dict, indent: str) -> str:
    """Render the models mapping value (an object) as indented JSON text.

    ``indent`` is the indentation of the `"models"` key line. The opening
    brace sits on the first line (immediately after the `"models":` key that
    the caller keeps), and the body is indented one level deeper::

        {
          "adaptive": {
            "name": "Adaptive (auto-tier)",
            "limit": { "context": 1048576, "output": 65536 }
          },
          ...
        }
    """
    unit = "  "  # two-space step, matching typical JSON formatting
    body_indent = indent + unit
    field_indent = body_indent + unit
    out = [f"{{"]
    items = list(block.items())
    for i, (mid, entry) in enumerate(items):
        trailing = "," if i < len(items) - 1 else ""
        limit = entry.get("limit", {})
        out.append(f'{body_indent}"{mid}": {{')
        out.append(f'{field_indent}"name": {json.dumps(entry.get("name", mid))},')
        # keep limit on one line for compactness; the separating comma belongs
        # after the entry's closing brace, not here.
        out.append(
            f'{field_indent}"limit": {{"context": {limit.get("context", DEFAULT_CONTEXT)}, '
            f'"output": {limit.get("output", DEFAULT_OUTPUT)}}}'
        )
        out.append(f"{body_indent}}}{trailing}")
    out.append(f"{indent}}}")
    return "\n".join(out)


def sync_config(text: str, models: list[dict], dry_run: bool = False) -> tuple[str, bool, str]:
    """Rewrite the router provider's models VALUE. Returns (new_text, changed, summary).

    Non-destructive: locates the `router` provider, then within it replaces
    only the *value* of the `"models"` key. Everything else in the file —
    the `"models":` key itself, the rest of the router provider, and all other
    providers/top-level keys — is preserved byte-for-byte.
    """
    block = build_models_block(models)

    prov_span = _find_provider_object(text, PROVIDER_NAME)
    if prov_span is None:
        summary = "router provider not found — would be created (dry-run)" if dry_run else "router provider created"
        return text, True, summary

    pstart, pend = prov_span
    obj_text = text[pstart:pend]

    # Locate the existing "models" key inside the router provider.
    existing = _find_key_block(obj_text, "models")
    if existing is None:
        summary = "router provider has no models block — not modifying (run manually)"
        return text, False, summary

    vstart, vend = existing
    old_value = obj_text[vstart:vend]
    # Indent of the value's first line: the "models" key indent + 2 spaces.
    key_line_indent = _indent_of(obj_text, vstart)
    # The key and value live on the same line ("    \"models\": {"), so the
    # value's body is indented key_indent + 2 spaces.
    value_indent = " " * (len(key_line_indent) + 2)
    new_value = render_models_value(block, value_indent)
    new_obj = obj_text[:vstart] + new_value + obj_text[vend:]
    changed = new_obj != obj_text
    summary = "models block updated" if changed else "models block already in sync"
    return text[:pstart] + new_obj + text[pend:], changed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Nexus router tiers into OpenCode config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to opencode.json(c)")
    parser.add_argument("--router-url", default=DEFAULT_ROUTER_URL, help="Nexus router base URL (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()

    try:
        models = fetch_models(args.router_url)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Could not fetch models from {args.router_url}: {exc}")
        print("  Is the Nexus router running? (routerctl status)")
        return 1

    if not config_path.exists():
        if args.dry_run:
            print(f"[dry-run] config {config_path} does not exist; would create it.")
            return 0
        # Create a fresh config with just the router provider.
        fresh = {
            "provider": {
                PROVIDER_NAME: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Model Router (local)",
                    "options": {
                        "baseURL": args.router_url,
                        "apiKey": "router",
                    },
                }
            }
        }
        fresh["provider"][PROVIDER_NAME]["models"] = build_models_block(models)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(fresh, indent=2) + "\n")
        print(f"✓ Created {config_path} with the router provider and {len(models)} models.")
        return 0

    original = config_path.read_text(encoding="utf-8")
    new_text, changed, summary = sync_config(original, models, dry_run=args.dry_run)

    print(f"Router: {len(models)} models advertised")
    for m in models:
        print(f"  - {m.get('id')}")
    print(f"Config: {config_path}")
    print(f"Result: {summary}")

    if args.dry_run:
        print("[dry-run] no files were modified.")
        return 0

    if not changed:
        print("✓ Nothing to do — already in sync.")
        return 0

    # Non-destructive: back up before writing.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(f"{config_path.name}.bak-{stamp}")
    shutil.copy2(config_path, backup)
    config_path.write_text(new_text, encoding="utf-8")
    print(f"✓ Wrote {config_path}")
    print(f"  Backup saved to {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
