.PHONY: run setup setup-dev ruff clean

setup:
	uv sync --no-dev

setup-dev:
	uv sync

run:
	uv run python -m src.main

ruff:
	uv run ruff check src --fix
	uv run ruff format src
