"""Model validation module for PyTorch Mobile."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


logger = logging.getLogger(__name__)


def load_torchscript_model(model_path: Path) -> torch.jit.ScriptModule:
    """Load a TorchScript model.

    Args:
        model_path: Path to the TorchScript model

    Returns:
        Loaded TorchScript model

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        logger.info(f"Loading model from: {model_path}")
        model = torch.jit.load(str(model_path))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


def preprocess_image(image_path: Path, size: int = 320) -> torch.Tensor:
    """Preprocess an image for model inference.

    Args:
        image_path: Path to the image file
        size: Target size for the image

    Returns:
        Preprocessed image tensor

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(
    model: torch.jit.ScriptModule,
    image_tensor: torch.Tensor,
    top_k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference on an image tensor.

    Args:
        model: TorchScript model
        image_tensor: Preprocessed image tensor
        top_k: Number of top predictions to return

    Returns:
        Tuple of (probabilities, class_indices)
    """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    return top_probs, top_indices


def validate_model(
    model_path: Path,
    image_path: Optional[Path] = None,
    class_names: Optional[list[str]] = None,
    top_k: int = 5,
) -> dict[str, any]:
    """Validate a TorchScript model on an image.

    Args:
        model_path: Path to the TorchScript model
        image_path: Path to test image (optional)
        class_names: List of class names (optional)
        top_k: Number of top predictions to return

    Returns:
        Dictionary with validation results
    """
    # Load model
    model = load_torchscript_model(model_path)

    # Get model info
    file_size = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {file_size:.2f} MB")

    results = {
        "model_path": str(model_path),
        "model_size_mb": file_size,
        "success": True,
    }

    # If image provided, run inference
    if image_path:
        logger.info(f"Running inference on: {image_path}")
        image_tensor = preprocess_image(image_path)

        # Benchmark inference time
        import time

        start_time = time.time()
        top_probs, top_indices = predict(model, image_tensor, top_k)
        inference_time = (time.time() - start_time) * 1000

        logger.info(f"Inference time: {inference_time:.2f} ms")

        # Format predictions
        predictions = []
        for i in range(top_k):
            prob = top_probs[i].item()
            idx = top_indices[i].item()
            class_name = class_names[idx] if class_names else f"Class {idx}"

            predictions.append(
                {
                    "class": class_name,
                    "index": idx,
                    "probability": prob,
                    "confidence_percent": prob * 100,
                }
            )

            logger.info(f"  {i + 1}. {class_name}: {prob * 100:.2f}%")

        results.update(
            {
                "inference_time_ms": inference_time,
                "predictions": predictions,
            }
        )

    return results


def compare_models(
    model_paths: list[Path],
    image_path: Path,
    class_names: Optional[list[str]] = None,
) -> dict[str, any]:
    """Compare multiple models on the same image.

    Args:
        model_paths: List of model paths to compare
        image_path: Test image path
        class_names: List of class names (optional)

    Returns:
        Comparison results
    """
    logger.info(f"Comparing {len(model_paths)} models")

    image_tensor = preprocess_image(image_path)
    results = {"image_path": str(image_path), "models": []}

    for model_path in model_paths:
        logger.info(f"\nTesting: {model_path}")
        model_result = validate_model(model_path, image_path, class_names)
        results["models"].append(model_result)

    return results


def verify_android_compatibility(model_path: Path) -> dict[str, any]:
    """Verify that a model is compatible with Android deployment.

    Args:
        model_path: Path to the model

    Returns:
        Compatibility check results
    """
    logger.info("Verifying Android compatibility...")

    checks = {
        "file_exists": model_path.exists(),
        "file_extension": model_path.suffix in [".pt", ".ptl"],
        "loadable": False,
        "traced": False,
        "optimized": False,
    }

    if checks["file_exists"]:
        try:
            model = torch.jit.load(str(model_path))
            checks["loadable"] = True

            # Check if model was traced (vs scripted)
            # Traced models are generally more compatible
            try:
                # Try to access the graph - traced models have this
                _ = model.graph
                checks["traced"] = True
            except AttributeError:
                checks["traced"] = False

            # Check for mobile optimization
            # This is a heuristic - mobile optimized models are usually smaller
            file_size = model_path.stat().st_size / (1024 * 1024)
            if file_size < 50:  # Less than 50MB suggests optimization
                checks["optimized"] = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    all_passed = all(checks.values())
    checks["compatible"] = all_passed

    logger.info("Compatibility check results:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}")

    return checks


def main() -> None:
    """Main entry point for model validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate PyTorch Mobile model")
    parser.add_argument("--model", type=Path, required=True, help="Path to model file")
    parser.add_argument("--image", type=Path, help="Path to test image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    parser.add_argument(
        "--check-compatibility", action="store_true", help="Check Android compatibility"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.check_compatibility:
        verify_android_compatibility(args.model)

    if args.image:
        validate_model(args.model, args.image, top_k=args.top_k)
    else:
        # Just validate that model can be loaded
        validate_model(args.model)


if __name__ == "__main__":
    main()
