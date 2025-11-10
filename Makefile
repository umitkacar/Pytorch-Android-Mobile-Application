.PHONY: help install install-dev clean lint format type-check test test-cov pre-commit-install pre-commit-run build export-model quick-export

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(BLUE)PyTorch Android Mobile Application - Makefile Commands$(NC)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Installation & Setup
# ============================================================================

install: ## Install dependencies
	@echo '$(BLUE)Installing dependencies...$(NC)'
	pip install -e .

install-dev: ## Install development dependencies
	@echo '$(BLUE)Installing development dependencies...$(NC)'
	pip install -e ".[dev]"

install-all: ## Install all dependencies (dev + training)
	@echo '$(BLUE)Installing all dependencies...$(NC)'
	pip install -e ".[all]"

pre-commit-install: ## Install pre-commit hooks
	@echo '$(BLUE)Installing pre-commit hooks...$(NC)'
	pre-commit install

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linter (ruff)
	@echo '$(BLUE)Running linter...$(NC)'
	ruff check python/src python/tests

lint-fix: ## Run linter with auto-fix
	@echo '$(BLUE)Running linter with auto-fix...$(NC)'
	ruff check --fix python/src python/tests

format: ## Format code with black
	@echo '$(BLUE)Formatting code...$(NC)'
	black python/src python/tests

format-check: ## Check code formatting without changes
	@echo '$(BLUE)Checking code formatting...$(NC)'
	black --check python/src python/tests

type-check: ## Run type checker (mypy)
	@echo '$(BLUE)Running type checker...$(NC)'
	mypy python/src

pre-commit-run: ## Run all pre-commit hooks
	@echo '$(BLUE)Running pre-commit hooks...$(NC)'
	pre-commit run --all-files

check-all: format lint type-check ## Run all code quality checks

# ============================================================================
# Testing
# ============================================================================

test: ## Run tests
	@echo '$(BLUE)Running tests...$(NC)'
	pytest python/tests -v

test-parallel: ## Run tests in parallel with pytest-xdist
	@echo '$(BLUE)Running tests in parallel...$(NC)'
	pytest python/tests -n auto -v

test-cov: ## Run tests with coverage
	@echo '$(BLUE)Running tests with coverage...$(NC)'
	pytest python/tests --cov=pytorch_mobile --cov-report=html --cov-report=term --cov-report=xml -v

test-fast: ## Run tests without slow tests
	@echo '$(BLUE)Running fast tests...$(NC)'
	pytest python/tests -n auto -v -m "not slow"

test-watch: ## Run tests in watch mode
	@echo '$(BLUE)Running tests in watch mode...$(NC)'
	pytest-watch python/tests -v

# ============================================================================
# Model Training & Export
# ============================================================================

export-model: ## Export pretrained model to TorchScript
	@echo '$(BLUE)Exporting pretrained model...$(NC)'
	python -m pytorch_mobile.export --model mobilenet_v2 --output models/model.pt --optimize --benchmark

quick-export: ## Quick export pretrained model and copy to Android assets
	@echo '$(BLUE)Quick export to Android...$(NC)'
	bash python/scripts/quick_export.sh

train-model: ## Train a model (requires data directory)
	@echo '$(BLUE)Training model...$(NC)'
	@echo '$(YELLOW)Note: You need to provide --data-dir argument$(NC)'
	python -m pytorch_mobile.train --help

validate-model: ## Validate exported model
	@echo '$(BLUE)Validating model...$(NC)'
	python -m pytorch_mobile.validate --model models/model.pt --check-compatibility

# ============================================================================
# Build & Package
# ============================================================================

build: ## Build Python package
	@echo '$(BLUE)Building package...$(NC)'
	hatch build

build-clean: clean build ## Clean and build package

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Clean build artifacts and caches
	@echo '$(BLUE)Cleaning build artifacts...$(NC)'
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

clean-all: clean ## Deep clean (including venv and models)
	@echo '$(BLUE)Deep cleaning...$(NC)'
	rm -rf .venv venv env
	rm -rf models/*.pt models/*.pth

# ============================================================================
# Android Development
# ============================================================================

android-build: ## Build Android app
	@echo '$(BLUE)Building Android app...$(NC)'
	cd HelloWorldApp && ./gradlew build

android-install: ## Install Android app to connected device
	@echo '$(BLUE)Installing Android app...$(NC)'
	cd HelloWorldApp && ./gradlew installDebug

android-run: ## Run Android app on connected device
	@echo '$(BLUE)Running Android app...$(NC)'
	cd HelloWorldApp && ./gradlew installDebug
	adb shell am start -n org.pytorch.helloworld/.MainActivity

# ============================================================================
# Development Workflow
# ============================================================================

dev-setup: install-dev pre-commit-install ## Complete development setup
	@echo '$(GREEN)✅ Development environment ready!$(NC)'

dev-check: format lint type-check test ## Run all checks before commit
	@echo '$(GREEN)✅ All checks passed!$(NC)'

ci: format-check lint type-check test-cov ## Run CI checks
	@echo '$(GREEN)✅ CI checks passed!$(NC)'

# ============================================================================
# Docker (optional)
# ============================================================================

docker-build: ## Build Docker image
	@echo '$(BLUE)Building Docker image...$(NC)'
	docker build -t pytorch-mobile-trainer .

docker-run: ## Run in Docker container
	@echo '$(BLUE)Running in Docker container...$(NC)'
	docker run -it --rm -v $(PWD):/workspace pytorch-mobile-trainer

# ============================================================================
# Documentation
# ============================================================================

docs-serve: ## Serve documentation locally
	@echo '$(BLUE)Serving documentation...$(NC)'
	@echo '$(YELLOW)Note: Install docs dependencies first: pip install -e ".[docs]"$(NC)'
	mkdocs serve

docs-build: ## Build documentation
	@echo '$(BLUE)Building documentation...$(NC)'
	mkdocs build

# ============================================================================
# Info
# ============================================================================

info: ## Show project information
	@echo '$(BLUE)Project Information$(NC)'
	@echo '==================='
	@echo 'Project: PyTorch Android Mobile Application'
	@echo 'Python Version:' $(shell python --version)
	@echo 'PyTorch Version:' $(shell python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
	@echo 'Hatch Version:' $(shell hatch --version 2>/dev/null || echo "Not installed")
	@echo ''
	@echo 'Dependencies Status:'
	@pip list | grep -E "(torch|pytest|ruff|black|mypy|hatch)" || echo "Dependencies not installed"
