"""PyTorch Mobile Trainer - Model training and export toolkit for Android deployment.

This package provides utilities for training, optimizing, and exporting PyTorch models
for deployment on Android devices using PyTorch Mobile.
"""

__version__ = "1.0.0"
__author__ = "Ümit Kacar"
__email__ = "umitkacar@users.noreply.github.com"

from pytorch_mobile.export import export_to_torchscript
from pytorch_mobile.train import train_model
from pytorch_mobile.validate import validate_model


__all__ = [
    "train_model",
    "export_to_torchscript",
    "validate_model",
    "__version__",
]
