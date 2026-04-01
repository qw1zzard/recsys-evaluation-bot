.PHONY: run dev format lint setup

setup:
	uv sync

run:
	uv run python -m src.main

format:
	uv run black src
	uv run isort src

lint:
	uv run flake8 src
	uv run pylint src
