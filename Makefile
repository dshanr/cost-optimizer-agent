.PHONY: install test test-live eval lint format check clean run demo langfuse langfuse-down

install:
	uv sync --extra dev

test:
	uv run pytest -m "not live"

test-live:
	uv run pytest -m live

eval:
	uv run python -m evals.eval_runner

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

check: lint
	uv run mypy src/cost_optimizer

run:
	uv run cost-optimizer run data/sample_aws_cur.csv

demo:
	uv run streamlit run app/streamlit_app.py

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info

langfuse:
	docker compose up -d
	@echo "Langfuse: http://localhost:3000  (dev@local / dev-password)"

langfuse-down:
	docker compose down
