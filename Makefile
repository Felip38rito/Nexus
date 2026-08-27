# Makefile for Nexus model router
# Provides common development and management commands.

.PHONY: install test restart status logs clean setup-dev

install:
	bash install.sh

test:
	uv run pytest

restart:
	./nexusctl restart

status:
	./nexusctl status

logs:
	./nexusctl logs

clean:
	rm -f logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} +

setup-dev:
	uv sync --extra dev
