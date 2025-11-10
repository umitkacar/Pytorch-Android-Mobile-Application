#!/bin/bash
# Example script for exporting a model to TorchScript

set -e

echo "📦 Exporting PyTorch model to TorchScript"
echo "=========================================="

# Configuration
MODEL_NAME="mobilenet_v2"
WEIGHTS_PATH="./models/${MODEL_NAME}_best.pth"
OUTPUT_PATH="./models/model.pt"
NUM_CLASSES=1000

# Export model
python -m pytorch_mobile.export \
    --model "$MODEL_NAME" \
    --weights "$WEIGHTS_PATH" \
    --output "$OUTPUT_PATH" \
    --num-classes "$NUM_CLASSES" \
    --optimize \
    --benchmark

echo "✅ Export complete!"
echo "Model exported to: $OUTPUT_PATH"
echo ""
echo "📱 To use in Android app:"
echo "   1. Copy $OUTPUT_PATH to HelloWorldApp/app/src/main/assets/model.pt"
echo "   2. Build and run the Android app"
