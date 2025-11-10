"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest
import torch

from PIL import Image
from torch import nn


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for models.

    Args:
        tmp_path: Pytest temporary path fixture

    Yields:
        Path to temporary model directory
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test data.

    Args:
        tmp_path: Pytest temporary path fixture

    Yields:
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_model() -> nn.Module:
    """Create a simple test model.

    Returns:
        Simple PyTorch model for testing
    """

    class SimpleModel(nn.Module):
        """Simple CNN model for testing."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def sample_image(temp_data_dir: Path) -> Path:
    """Create a sample test image.

    Args:
        temp_data_dir: Temporary data directory

    Returns:
        Path to sample image
    """
    image_path = temp_data_dir / "test_image.jpg"
    image = Image.new("RGB", (320, 320), color=(100, 150, 200))
    image.save(image_path)
    return image_path


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample input tensor.

    Returns:
        Sample tensor for testing
    """
    return torch.rand(1, 3, 320, 320)


@pytest.fixture
def class_names() -> list[str]:
    """Get sample class names.

    Returns:
        List of class names
    """
    return [f"class_{i}" for i in range(10)]
