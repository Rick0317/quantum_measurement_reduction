# Development Setup

This project includes comprehensive CI/CD setup with automated linting, formatting, and testing.

## Quick Start

### Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### Auto-format Code
```bash
# Format all Python files
make format

# Or manually:
black .
isort .
```

### Run All Checks Locally
```bash
make ci
```

## Available Commands

Use `make help` to see all available commands:

- `make install` - Install the package
- `make install-dev` - Install with development dependencies
- `make format` - Auto-format code with black and isort
- `make lint` - Run linting checks (flake8, black, isort)
- `make type-check` - Run type checking with mypy
- `make test` - Run tests
- `make security-check` - Run security checks with bandit
- `make clean` - Clean build artifacts

## Pre-commit Hooks

Install pre-commit hooks to automatically format code before commits:

```bash
make pre-commit-install
```

This will:
- Format code with black and isort
- Run flake8 linting
- Run mypy type checking
- Run security checks with bandit
- Check for common issues (trailing whitespace, large files, etc.)

## GitHub Actions CI

The project includes GitHub Actions workflows that run on every push and pull request:

- **Linting**: flake8, black, isort
- **Type Checking**: mypy
- **Testing**: pytest with coverage
- **Security**: bandit
- **Multi-Python Support**: Tests on Python 3.8, 3.9, 3.10, 3.11

## Configuration Files

- `.github/workflows/ci.yml` - GitHub Actions CI configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `pyproject.toml` - Tool configurations (black, isort, flake8, mypy, pytest)
- `Makefile` - Convenient commands for development
- `.gitignore` - Comprehensive Python gitignore

## Code Style

The project uses:
- **Black** for code formatting (88 character line length)
- **isort** for import sorting (compatible with black)
- **flake8** for linting
- **mypy** for type checking
- **bandit** for security analysis

All tools are configured in `pyproject.toml` and can be customized as needed.
