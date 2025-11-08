# 🐍 PyTorch Mobile Trainer

Python toolkit for training, optimizing, and exporting PyTorch models for Android deployment.

## 🚀 Quick Start

### Installation

```bash
# Install package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install everything (dev + training tools)
pip install -e ".[all]"
```

### Quick Export (Pretrained Model)

The fastest way to get a model for your Android app:

```bash
# Export pretrained MobileNetV2 and copy to Android assets
make quick-export

# Or manually:
python -m pytorch_mobile.export \
    --model mobilenet_v2 \
    --output models/model.pt \
    --optimize \
    --quantize
```

This will:
1. ✅ Download a pretrained model
2. ✅ Optimize for mobile
3. ✅ Apply quantization (smaller size)
4. ✅ Benchmark performance
5. ✅ Copy to Android assets (if available)

## 📦 Features

### Model Training

Train custom models for your Android app:

```bash
python -m pytorch_mobile.train \
    --model mobilenet_v2 \
    --data-dir ./data \
    --output-dir ./models \
    --epochs 10 \
    --batch-size 32
```

**Supported Models:**
- `mobilenet_v2` - Lightweight, fast (recommended for mobile)
- `resnet18` - Good accuracy, moderate size
- `resnet34` - Better accuracy, larger size

### Model Export

Export trained models to TorchScript format:

```bash
python -m pytorch_mobile.export \
    --model mobilenet_v2 \
    --weights models/mobilenet_v2_best.pth \
    --output models/model.pt \
    --optimize \
    --quantize \
    --benchmark
```

**Export Options:**
- `--optimize` - Optimize for mobile inference
- `--quantize` - Apply dynamic quantization (2-4x smaller)
- `--benchmark` - Measure inference performance
- `--format` - Export format (torchscript or onnx)

### Model Validation

Validate and test exported models:

```bash
# Check Android compatibility
python -m pytorch_mobile.validate \
    --model models/model.pt \
    --check-compatibility

# Test inference on image
python -m pytorch_mobile.validate \
    --model models/model.pt \
    --image test_image.jpg \
    --top-k 5
```

## 🛠️ Development Tools

### Modern Python Tooling

This project uses the latest 2024-2025 Python development tools:

#### **Hatch** - Modern Project Manager
```bash
# Create development environment
hatch shell

# Run tests
hatch run test

# Run all checks
hatch run check-all
```

#### **Ruff** - Ultra-Fast Linter
```bash
# Lint code (replaces flake8, isort, pyupgrade, etc.)
ruff check python/src python/tests

# Auto-fix issues
ruff check --fix python/src python/tests
```

#### **Black** - Code Formatter
```bash
# Format code
black python/src python/tests

# Check formatting
black --check python/src python/tests
```

#### **MyPy** - Static Type Checker
```bash
# Type check code
mypy python/src
```

#### **Pytest** - Testing Framework
```bash
# Run tests
pytest python/tests

# Run with coverage
pytest python/tests --cov=pytorch_mobile --cov-report=html

# Run fast tests only
pytest python/tests -m "not slow"
```

### Pre-commit Hooks

Automatically check code quality before commits:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Hooks included:**
- ✅ Ruff (linting)
- ✅ Black (formatting)
- ✅ MyPy (type checking)
- ✅ Bandit (security checks)
- ✅ YAML/JSON validation
- ✅ Trailing whitespace removal
- ✅ And more!

## 📊 Code Quality Standards

- **Test Coverage:** >80% required
- **Type Hints:** Enforced with MyPy
- **Code Style:** Black (line length: 100)
- **Linting:** Ruff with comprehensive rule set
- **Documentation:** Docstrings required (interrogate check)

## 🔧 Makefile Commands

We provide a comprehensive Makefile for common tasks:

```bash
# Show all available commands
make help

# Development setup
make dev-setup              # Install deps + pre-commit hooks
make dev-check              # Run all checks before commit

# Code quality
make lint                   # Run ruff linter
make format                 # Format with black
make type-check             # Run mypy
make check-all              # Run all checks

# Testing
make test                   # Run tests
make test-cov               # Run tests with coverage
make test-fast              # Run fast tests only

# Model operations
make export-model           # Export pretrained model
make quick-export           # Quick export to Android
make validate-model         # Validate model

# Build & clean
make build                  # Build Python package
make clean                  # Clean artifacts
make clean-all              # Deep clean

# Android
make android-build          # Build Android app
make android-install        # Install to device
```

## 📁 Project Structure

```
python/
├── src/pytorch_mobile/      # Main package
│   ├── __init__.py          # Package initialization
│   ├── train.py             # Training module
│   ├── export.py            # Export module
│   └── validate.py          # Validation module
├── tests/                   # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── test_train.py        # Training tests
│   ├── test_export.py       # Export tests
│   └── test_validate.py     # Validation tests
├── scripts/                 # Utility scripts
│   ├── train_example.sh     # Training example
│   ├── export_example.sh    # Export example
│   └── quick_export.sh      # Quick export script
└── README.md                # This file
```

## 🎯 Typical Workflow

### 1. For Quick Start (Pretrained Model)

```bash
# Export pretrained model and copy to Android
make quick-export

# Validate
make validate-model

# Build Android app
make android-build
```

### 2. For Custom Training

```bash
# Prepare your data
# data/
#   ├── train/
#   │   ├── class1/
#   │   └── class2/
#   └── val/
#       ├── class1/
#       └── class2/

# Train model
python -m pytorch_mobile.train \
    --model mobilenet_v2 \
    --data-dir ./data \
    --epochs 20

# Export trained model
python -m pytorch_mobile.export \
    --model mobilenet_v2 \
    --weights models/mobilenet_v2_best.pth \
    --output models/model.pt \
    --optimize \
    --quantize

# Copy to Android
cp models/model.pt ../HelloWorldApp/app/src/main/assets/
```

### 3. Development Workflow

```bash
# Setup development environment
make dev-setup

# Make changes to code...

# Check code quality
make dev-check

# Run tests
make test-cov

# Commit (pre-commit hooks will run automatically)
git commit -m "your changes"
```

## 🔬 Testing

### Run Tests

```bash
# All tests
pytest python/tests -v

# With coverage report
pytest python/tests --cov=pytorch_mobile --cov-report=html

# Specific test file
pytest python/tests/test_export.py -v

# Specific test
pytest python/tests/test_export.py::TestExportToTorchScript::test_export_basic_model -v
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run GPU tests (requires GPU)
pytest -m gpu
```

## 📊 Coverage Reports

After running tests with coverage:

```bash
# View HTML report
open htmlcov/index.html

# View terminal report
pytest --cov=pytorch_mobile --cov-report=term
```

## 🤝 Contributing

Before committing:

```bash
# Run all checks
make dev-check

# Or manually:
make format      # Format code
make lint        # Check linting
make type-check  # Check types
make test-cov    # Run tests with coverage
```

## 📚 API Reference

See [docs/API.md](../docs/API.md) for detailed API documentation.

## 🐛 Troubleshooting

### Import Errors

```bash
# Make sure package is installed in editable mode
pip install -e .
```

### PyTorch Not Found

```bash
# Install PyTorch (CPU version)
pip install torch torchvision

# Or with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Pre-commit Hooks Failing

```bash
# Update hooks
pre-commit autoupdate

# Clear cache and retry
pre-commit clean
pre-commit run --all-files
```

## 📖 Resources

- [PyTorch Mobile Documentation](https://pytorch.org/mobile/)
- [TorchScript Guide](https://pytorch.org/docs/stable/jit.html)
- [Model Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## ⚖️ License

MIT License - see [LICENSE](../LICENSE) for details.

---

**Made with ❤️ and PyTorch 🔥**
