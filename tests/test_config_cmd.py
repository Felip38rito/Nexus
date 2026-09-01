"""Tests for the axon CLI config commands (config_cmd.py)."""
from __future__ import annotations

from pathlib import Path

import pytest
import typer
import yaml

from axon_cli import config_cmd


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A valid config.yml in a temp dir, with config_path() pointed at it."""
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "default_tier": "air",
                "providers": {
                    "default": {
                        "base_url": "https://ollama.com/v1",
                        "api_key_env": "OLLAMA_API_KEY",
                    }
                },
                "tiers": {
                    "mini": {"model": "gemma4:31b", "name": "mini"},
                    "air": {"model": "deepseek-v4-flash:0731", "name": "air"},
                    "pro": {"model": "deepseek-v4-pro:0813", "name": "pro"},
                    "ultra": {"model": "kimi-k3", "name": "ultra"},
                },
                "classifier": {"model": "gemma4:31b", "provider": "default"},
            },
            sort_keys=False,
        )
    )
    return path


@pytest.fixture
def patch_config_path(monkeypatch, config_file: Path):
    """Point config_path() at the temp config file."""
    monkeypatch.setattr(config_cmd, "config_path", lambda: config_file)
    return config_file


# --- _load_config -----------------------------------------------------------

def test_load_config_missing_raises(monkeypatch, tmp_path: Path):
    missing = tmp_path / "nope.yml"
    monkeypatch.setattr(config_cmd, "config_path", lambda: missing)
    with pytest.raises(typer.Exit) as exc:
        config_cmd._load_config()
    assert exc.value.exit_code == 1


def test_load_config_invalid_yaml_raises(monkeypatch, tmp_path: Path):
    bad = tmp_path / "bad.yml"
    bad.write_text("tiers: [unclosed")
    monkeypatch.setattr(config_cmd, "config_path", lambda: bad)
    with pytest.raises(typer.Exit) as exc:
        config_cmd._load_config()
    assert exc.value.exit_code == 1


def test_load_config_not_mapping_raises(monkeypatch, tmp_path: Path):
    not_map = tmp_path / "list.yml"
    not_map.write_text("- a\n- b\n")
    monkeypatch.setattr(config_cmd, "config_path", lambda: not_map)
    with pytest.raises(typer.Exit) as exc:
        config_cmd._load_config()
    assert exc.value.exit_code == 1


def test_load_config_ok(patch_config_path, config_file: Path):
    data = config_cmd._load_config()
    assert data["tiers"]["pro"]["model"] == "deepseek-v4-pro:0813"


# --- _save_config -----------------------------------------------------------

def test_save_config_writes_file(tmp_path: Path):
    target = tmp_path / "sub" / "config.yml"
    config_cmd._save_config({"a": 1}, path=target)
    assert target.exists()
    assert yaml.safe_load(target.read_text()) == {"a": 1}


# --- list_config ------------------------------------------------------------

def test_list_config_prints_table(patch_config_path, capsys):
    config_cmd.list_config()
    out = capsys.readouterr().out
    assert "Axon Config" in out
    assert "deepseek-v4-pro:0813" in out
    assert "kimi-k3" in out


# --- set_tier_model ---------------------------------------------------------

def test_set_tier_model_updates(patch_config_path, config_file: Path):
    config_cmd.set_tier_model("pro", "new-pro-model")
    data = yaml.safe_load(config_file.read_text())
    assert data["tiers"]["pro"]["model"] == "new-pro-model"


def test_set_tier_model_unknown_tier_raises(patch_config_path):
    with pytest.raises(typer.Exit) as exc:
        config_cmd.set_tier_model("bogus", "x")
    assert exc.value.exit_code == 1
