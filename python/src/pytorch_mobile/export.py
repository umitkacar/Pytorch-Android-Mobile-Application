"""Model export module for PyTorch Mobile."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile


logger = logging.getLogger(__name__)


def export_to_torchscript(
    model: nn.Module,
    output_path: Path,
    input_size: Tuple[int, int, int, int] = (1, 3, 320, 320),
    optimize: bool = True,
    quantize: bool = False,
) -> Path:
    """Export PyTorch model to TorchScript format for mobile deployment.

    Args:
        model: PyTorch model to export
        output_path: Path to save the exported model
        input_size: Input tensor size (batch, channels, height, width)
        optimize: Whether to optimize for mobile
        quantize: Whether to apply dynamic quantization

    Returns:
        Path to the exported model

    Raises:
        RuntimeError: If export fails
    """
    model.eval()

    # Create example input
    example_input = torch.rand(input_size)
    logger.info(f"Using example input shape: {example_input.shape}")

    try:
        # Trace the model
        logger.info("Tracing model...")
        traced_model = torch.jit.trace(model, example_input)

        # Apply quantization if requested
        if quantize:
            logger.info("Applying dynamic quantization...")
            traced_model = torch.quantization.quantize_dynamic(
                traced_model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8,
            )

        # Optimize for mobile if requested
        if optimize:
            logger.info("Optimizing for mobile...")
            traced_model = optimize_for_mobile(traced_model)

        # Save the model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced_model._save_for_lite_interpreter(str(output_path))

        # Verify the saved model
        file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"Model exported successfully to: {output_path}")
        logger.info(f"Model size: {file_size:.2f} MB")

        return output_path

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise RuntimeError(f"Model export failed: {e}") from e


def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: Tuple[int, int, int, int] = (1, 3, 320, 320),
    opset_version: int = 11,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save the ONNX model
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version

    Returns:
        Path to the exported ONNX model

    Raises:
        RuntimeError: If export fails
    """
    model.eval()

    example_input = torch.rand(input_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Exporting to ONNX with opset version {opset_version}...")
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model exported successfully to: {output_path}")
        logger.info(f"Model size: {file_size:.2f} MB")

        return output_path

    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        raise RuntimeError(f"ONNX export failed: {e}") from e


def optimize_model(
    model: nn.Module,
    quantize: bool = True,
    prune: bool = False,
    pruning_amount: float = 0.3,
) -> nn.Module:
    """Optimize model for mobile deployment.

    Args:
        model: PyTorch model to optimize
        quantize: Whether to apply quantization
        prune: Whether to apply pruning
        pruning_amount: Amount of weights to prune (0.0-1.0)

    Returns:
        Optimized PyTorch model
    """
    model.eval()

    if quantize:
        logger.info("Applying quantization...")
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8,
        )

    if prune:
        logger.info(f"Applying pruning (amount: {pruning_amount})...")
        # Import pruning utilities
        import torch.nn.utils.prune as prune_utils

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune_utils.l1_unstructured(module, name="weight", amount=pruning_amount)
                prune_utils.remove(module, "weight")

    return model


def benchmark_model(
    model_path: Path,
    input_size: Tuple[int, int, int, int] = (1, 3, 320, 320),
    num_iterations: int = 100,
) -> dict[str, float]:
    """Benchmark exported model performance.

    Args:
        model_path: Path to the exported model
        input_size: Input tensor size
        num_iterations: Number of iterations for benchmarking

    Returns:
        Dictionary with benchmark results
    """
    import time

    logger.info(f"Loading model from: {model_path}")
    model = torch.jit.load(str(model_path))
    model.eval()

    example_input = torch.rand(input_size)

    # Warmup
    logger.info("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(example_input)

    # Benchmark
    logger.info(f"Running benchmark ({num_iterations} iterations)...")
    timings = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(example_input)
            end_time = time.time()
            timings.append(end_time - start_time)

    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)

    results = {
        "avg_inference_time_ms": avg_time * 1000,
        "min_inference_time_ms": min_time * 1000,
        "max_inference_time_ms": max_time * 1000,
        "fps": 1.0 / avg_time,
    }

    logger.info("Benchmark results:")
    logger.info(f"  Average inference time: {results['avg_inference_time_ms']:.2f} ms")
    logger.info(f"  Min inference time: {results['min_inference_time_ms']:.2f} ms")
    logger.info(f"  Max inference time: {results['max_inference_time_ms']:.2f} ms")
    logger.info(f"  FPS: {results['fps']:.2f}")

    return results


def main() -> None:
    """Main entry point for model export."""
    import argparse

    from pytorch_mobile.train import create_model

    parser = argparse.ArgumentParser(description="Export PyTorch model for mobile deployment")
    parser.add_argument("--model", default="mobilenet_v2", help="Model architecture")
    parser.add_argument(
        "--weights", type=Path, required=False, help="Path to model weights (.pth file)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/model.pt"),
        help="Output path for exported model",
    )
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument(
        "--optimize", action="store_true", default=True, help="Optimize for mobile"
    )
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the exported model")
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create model
    model = create_model(args.model, args.num_classes, pretrained=True)

    # Load weights if provided
    if args.weights:
        logger.info(f"Loading weights from: {args.weights}")
        model.load_state_dict(torch.load(args.weights))

    # Export model
    if args.format == "torchscript":
        export_to_torchscript(
            model,
            args.output,
            optimize=args.optimize,
            quantize=args.quantize,
        )
    else:
        export_onnx(model, args.output)

    # Benchmark if requested
    if args.benchmark and args.format == "torchscript":
        benchmark_model(args.output)


if __name__ == "__main__":
    main()
