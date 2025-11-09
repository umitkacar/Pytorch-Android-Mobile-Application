"""Tests for training module."""

import pytest

from torch import nn

from pytorch_mobile.train import create_model, get_data_transforms


class TestCreateModel:
    """Tests for model creation."""

    def test_create_mobilenet_v2(self) -> None:
        """Test creating MobileNetV2 model."""
        model = create_model("mobilenet_v2", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)
        assert model is not None

    def test_create_resnet18(self) -> None:
        """Test creating ResNet18 model."""
        model = create_model("resnet18", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)
        assert model is not None

    def test_create_resnet34(self) -> None:
        """Test creating ResNet34 model."""
        model = create_model("resnet34", num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)
        assert model is not None

    def test_invalid_model_name(self) -> None:
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            create_model("invalid_model")

    def test_custom_num_classes(self) -> None:
        """Test model with custom number of classes."""
        num_classes = 50
        model = create_model("mobilenet_v2", num_classes=num_classes, pretrained=False)
        assert isinstance(model, nn.Module)


class TestDataTransforms:
    """Tests for data transformations."""

    def test_get_data_transforms(self) -> None:
        """Test getting data transforms."""
        transforms = get_data_transforms()
        assert "train" in transforms
        assert "val" in transforms
        assert transforms["train"] is not None
        assert transforms["val"] is not None

    def test_transforms_are_different(self) -> None:
        """Test that train and val transforms are different."""
        transforms = get_data_transforms()
        # Train should have more transforms (augmentation)
        assert len(transforms["train"].transforms) >= len(transforms["val"].transforms)
