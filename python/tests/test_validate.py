"""Tests for validation module."""

from pathlib import Path

import pytest
import torch

from pytorch_mobile.export import export_to_torchscript
from pytorch_mobile.validate import (
    load_torchscript_model,
    predict,
    preprocess_image,
    validate_model,
    verify_android_compatibility,
)


class TestLoadTorchScriptModel:
    """Tests for loading TorchScript models."""

    def test_load_valid_model(self, sample_model, temp_model_dir: Path) -> None:
        """Test loading a valid TorchScript model."""
        model_path = temp_model_dir / "model.pt"
        export_to_torchscript(sample_model, model_path)

        loaded_model = load_torchscript_model(model_path)
        assert loaded_model is not None

    def test_load_nonexistent_model(self, temp_model_dir: Path) -> None:
        """Test loading a nonexistent model raises error."""
        model_path = temp_model_dir / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            load_torchscript_model(model_path)


class TestPreprocessImage:
    """Tests for image preprocessing."""

    def test_preprocess_valid_image(self, sample_image: Path) -> None:
        """Test preprocessing a valid image."""
        tensor = preprocess_image(sample_image)

        assert tensor is not None
        assert tensor.dim() == 4  # (batch, channels, height, width)
        assert tensor.shape[0] == 1  # Batch size
        assert tensor.shape[1] == 3  # RGB channels

    def test_preprocess_nonexistent_image(self, temp_data_dir: Path) -> None:
        """Test preprocessing nonexistent image raises error."""
        image_path = temp_data_dir / "nonexistent.jpg"
        with pytest.raises(FileNotFoundError):
            preprocess_image(image_path)

    def test_preprocess_custom_size(self, sample_image: Path) -> None:
        """Test preprocessing with custom size."""
        size = 224
        tensor = preprocess_image(sample_image, size=size)

        assert tensor.shape[2] == size
        assert tensor.shape[3] == size


class TestPredict:
    """Tests for prediction."""

    def test_predict_returns_results(
        self, sample_model, temp_model_dir: Path, sample_tensor: torch.Tensor
    ) -> None:
        """Test that predict returns probabilities and indices."""
        model_path = temp_model_dir / "model.pt"
        export_to_torchscript(sample_model, model_path)

        model = load_torchscript_model(model_path)
        top_probs, top_indices = predict(model, sample_tensor, top_k=5)

        assert len(top_probs) == 5
        assert len(top_indices) == 5
        assert all(0 <= prob <= 1 for prob in top_probs)


class TestValidateModel:
    """Tests for model validation."""

    def test_validate_model_without_image(
        self, sample_model, temp_model_dir: Path
    ) -> None:
        """Test validating model without test image."""
        model_path = temp_model_dir / "model.pt"
        export_to_torchscript(sample_model, model_path)

        results = validate_model(model_path)

        assert results["success"] is True
        assert results["model_size_mb"] > 0
        assert "predictions" not in results

    def test_validate_model_with_image(
        self, sample_model, temp_model_dir: Path, sample_image: Path, class_names: list[str]
    ) -> None:
        """Test validating model with test image."""
        model_path = temp_model_dir / "model.pt"
        export_to_torchscript(sample_model, model_path)

        results = validate_model(
            model_path, image_path=sample_image, class_names=class_names, top_k=3
        )

        assert results["success"] is True
        assert "predictions" in results
        assert len(results["predictions"]) == 3
        assert "inference_time_ms" in results
        assert results["inference_time_ms"] > 0


class TestVerifyAndroidCompatibility:
    """Tests for Android compatibility verification."""

    def test_verify_valid_model(self, sample_model, temp_model_dir: Path) -> None:
        """Test verifying a valid model."""
        model_path = temp_model_dir / "model.pt"
        export_to_torchscript(sample_model, model_path)

        checks = verify_android_compatibility(model_path)

        assert checks["file_exists"] is True
        assert checks["loadable"] is True

    def test_verify_nonexistent_model(self, temp_model_dir: Path) -> None:
        """Test verifying nonexistent model."""
        model_path = temp_model_dir / "nonexistent.pt"
        checks = verify_android_compatibility(model_path)

        assert checks["file_exists"] is False
        assert checks["compatible"] is False
