# Makefile for Axon model router
# Provides common development and management commands.

.PHONY: install test restart status logs clean setup-dev

install:
	bash install.sh

test:
	uv run pytest

restart:
	./axonctl restart

status:
	./axonctl status

logs:
	./axonctl logs

clean:
	rm -f logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} +

setup-dev:
	uv sync --extra dev
