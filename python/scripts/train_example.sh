#!/bin/bash
# Example script for training a model

set -e

echo "🚀 Training PyTorch model for mobile deployment"
echo "================================================"

# Configuration
MODEL_NAME="mobilenet_v2"
DATA_DIR="./data"
OUTPUT_DIR="./models"
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_CLASSES=1000

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Train model
python -m pytorch_mobile.train \
    --model "$MODEL_NAME" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num-classes "$NUM_CLASSES"

echo "✅ Training complete!"
echo "Model saved to: $OUTPUT_DIR/${MODEL_NAME}_best.pth"
