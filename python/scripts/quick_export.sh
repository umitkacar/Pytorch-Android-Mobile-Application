#!/bin/bash
# Quick export of a pretrained model (no training required)

set -e

echo "⚡ Quick Export: Pretrained Model to TorchScript"
echo "================================================"

# Configuration
MODEL_NAME="${1:-mobilenet_v2}"
OUTPUT_PATH="./models/${MODEL_NAME}.pt"
ANDROID_ASSETS_PATH="../HelloWorldApp/app/src/main/assets/model.pt"

# Create models directory
mkdir -p ./models

# Export pretrained model
echo "Exporting pretrained $MODEL_NAME..."
python -m pytorch_mobile.export \
    --model "$MODEL_NAME" \
    --output "$OUTPUT_PATH" \
    --optimize \
    --quantize \
    --benchmark

# Copy to Android assets if directory exists
if [ -d "$(dirname "$ANDROID_ASSETS_PATH")" ]; then
    echo ""
    echo "📱 Copying to Android assets..."
    cp "$OUTPUT_PATH" "$ANDROID_ASSETS_PATH"
    echo "✅ Model copied to Android app!"
else
    echo ""
    echo "⚠️  Android assets directory not found."
    echo "   Manually copy $OUTPUT_PATH to your Android assets folder."
fi

echo ""
echo "✅ Quick export complete!"
echo "Model: $OUTPUT_PATH"
