.PHONY: help install install-dev test lint format format-check type-check security-check clean pre-commit-install

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting checks
	flake8 .
	black --check .
	isort --check-only .

format: ## Format code with black and isort
	black .
	isort .

format-check: ## Check code formatting
	black --check .
	isort --check-only .

type-check: ## Run type checking with mypy
	mypy . --ignore-missing-imports

security-check: ## Run security checks with bandit
	bandit -r . -f json -o bandit-report.json

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

ci: ## Run all CI checks locally
	make format-check
	make lint
	make type-check
	make test
