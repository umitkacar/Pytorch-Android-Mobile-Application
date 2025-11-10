"""Tests for export module."""

from pathlib import Path

import torch

from torch import nn

from pytorch_mobile.export import (
    benchmark_model,
    export_to_torchscript,
    optimize_model,
)


class TestExportToTorchScript:
    """Tests for TorchScript export."""

    def test_export_basic_model(self, sample_model: nn.Module, temp_model_dir: Path) -> None:
        """Test basic model export."""
        output_path = temp_model_dir / "model.pt"
        result_path = export_to_torchscript(
            sample_model, output_path, optimize=False, quantize=False
        )

        assert result_path.exists()
        assert result_path == output_path

    def test_export_with_optimization(self, sample_model: nn.Module, temp_model_dir: Path) -> None:
        """Test model export with mobile optimization."""
        output_path = temp_model_dir / "model_optimized.pt"
        result_path = export_to_torchscript(
            sample_model, output_path, optimize=True, quantize=False
        )

        assert result_path.exists()

    def test_export_with_quantization(self, sample_model: nn.Module, temp_model_dir: Path) -> None:
        """Test model export with quantization."""
        output_path = temp_model_dir / "model_quantized.pt"
        result_path = export_to_torchscript(
            sample_model, output_path, optimize=False, quantize=True
        )

        assert result_path.exists()

    def test_export_creates_directory(self, sample_model: nn.Module, temp_model_dir: Path) -> None:
        """Test that export creates parent directories."""
        output_path = temp_model_dir / "subdir" / "model.pt"
        result_path = export_to_torchscript(sample_model, output_path)

        assert result_path.exists()
        assert result_path.parent.exists()

    def test_exported_model_inference(
        self, sample_model: nn.Module, temp_model_dir: Path, sample_tensor: torch.Tensor
    ) -> None:
        """Test that exported model can run inference."""
        output_path = temp_model_dir / "model.pt"
        export_to_torchscript(sample_model, output_path)

        # Load and test
        loaded_model = torch.jit.load(str(output_path))
        output = loaded_model(sample_tensor)

        assert output is not None
        assert output.shape[0] == 1  # Batch size


class TestOptimizeModel:
    """Tests for model optimization."""

    def test_optimize_with_quantization(self, sample_model: nn.Module) -> None:
        """Test model optimization with quantization."""
        optimized = optimize_model(sample_model, quantize=True, prune=False)
        assert optimized is not None

    def test_optimize_without_quantization(self, sample_model: nn.Module) -> None:
        """Test model optimization without quantization."""
        optimized = optimize_model(sample_model, quantize=False, prune=False)
        assert optimized is not None


class TestBenchmarkModel:
    """Tests for model benchmarking."""

    def test_benchmark_exported_model(self, sample_model: nn.Module, temp_model_dir: Path) -> None:
        """Test benchmarking an exported model."""
        output_path = temp_model_dir / "model.pt"
        export_to_torchscript(sample_model, output_path)

        results = benchmark_model(output_path, num_iterations=10)

        assert "avg_inference_time_ms" in results
        assert "min_inference_time_ms" in results
        assert "max_inference_time_ms" in results
        assert "fps" in results
        assert results["avg_inference_time_ms"] > 0
        assert results["fps"] > 0
