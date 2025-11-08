"""Model training module for PyTorch Mobile."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


logger = logging.getLogger(__name__)


def create_model(
    model_name: str = "mobilenet_v2",
    num_classes: int = 1000,
    pretrained: bool = True,
) -> nn.Module:
    """Create a PyTorch model for training.

    Args:
        model_name: Name of the model architecture (mobilenet_v2, resnet18, etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        PyTorch model instance

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        if num_classes != 1000:
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        if num_classes != 1000:
            model.fc = nn.Linear(512, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        if num_classes != 1000:
            model.fc = nn.Linear(512, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_data_transforms() -> dict[str, transforms.Compose]:
    """Get data transformations for training and validation.

    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        progress_bar.set_postfix(
            {
                "loss": running_loss / total_samples,
                "acc": float(running_corrects) / total_samples,
            }
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = float(running_corrects) / total_samples

    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate for one epoch.

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation")
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        progress_bar.set_postfix(
            {
                "loss": running_loss / total_samples,
                "acc": float(running_corrects) / total_samples,
            }
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = float(running_corrects) / total_samples

    return epoch_loss, epoch_acc


def train_model(
    model_name: str = "mobilenet_v2",
    data_dir: Optional[Path] = None,
    output_dir: Path = Path("models"),
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 1000,
    pretrained: bool = True,
    device: Optional[str] = None,
) -> nn.Module:
    """Train a PyTorch model for mobile deployment.

    Args:
        model_name: Model architecture name
        data_dir: Directory containing training data
        output_dir: Directory to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to train on (cuda/cpu)

    Returns:
        Trained PyTorch model

    Raises:
        ValueError: If data_dir is None or doesn't exist
    """
    if data_dir is None:
        raise ValueError("data_dir must be provided")

    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    logger.info(f"Using device: {device_obj}")

    # Create model
    model = create_model(model_name, num_classes, pretrained)
    model = model.to(device_obj)

    # Data loading
    data_transforms = get_data_transforms()
    image_datasets = {
        x: datasets.ImageFolder(data_dir / x, data_transforms[x]) for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ["train", "val"]
    }

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model_path = output_dir / f"{model_name}_best.pth"

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 50)

        train_loss, train_acc = train_epoch(
            model, dataloaders["train"], criterion, optimizer, device_obj
        )
        val_loss, val_acc = validate_epoch(model, dataloaders["val"], criterion, device_obj)

        scheduler.step()

        logger.info(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with accuracy: {best_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")

    return model


def main() -> None:
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PyTorch model for mobile deployment")
    parser.add_argument("--model", default="mobilenet_v2", help="Model architecture")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
