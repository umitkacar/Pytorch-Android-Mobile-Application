# 📚 API Documentation

## Overview

This document provides detailed API documentation for key components in the PyTorch Android Mobile Application.

## 📱 MainActivity

The main activity that handles image selection and classification.

### Class: `MainActivity`

```java
public class MainActivity extends AppCompatActivity
```

#### Constants

```java
public static final int PICK_IMAGE = 1
```
Request code for image picker intent.

#### Fields

```java
Uri selectedImage
```
URI of the currently selected image.

```java
Module module = null
```
PyTorch module instance for model inference.

### Methods

#### `onCreate(Bundle savedInstanceState)`

```java
@Override
protected void onCreate(Bundle savedInstanceState)
```

**Description**: Initializes the activity and sets up UI components.

**Parameters**:
- `savedInstanceState` - Bundle containing saved state

**Behavior**:
- Sets up the content view
- Initializes button listeners
- Configures permission handlers

---

#### `assetFilePath(Context context, String assetName)`

```java
public static String assetFilePath(Context context, String assetName) throws IOException
```

**Description**: Copies an asset file to internal storage and returns its path.

**Parameters**:
- `context` - Application context
- `assetName` - Name of the asset file (e.g., "model.pt")

**Returns**: Absolute file path to the copied asset

**Throws**: `IOException` if file operations fail

**Example**:
```java
String modelPath = assetFilePath(this, "model.pt");
Module module = Module.load(modelPath);
```

---

#### `pickFromGallery()`

```java
private void pickFromGallery()
```

**Description**: Launches the system image picker.

**Behavior**:
- Creates an ACTION_PICK intent
- Filters for JPEG and PNG images
- Starts activity for result

**Example Flow**:
```
User taps button → pickFromGallery() → System Gallery → onActivityResult()
```

---

#### `onActivityResult(int requestCode, int resultCode, Intent data)`

```java
@Override
public void onActivityResult(int requestCode, int resultCode, Intent data)
```

**Description**: Handles the result from image picker and performs classification.

**Parameters**:
- `requestCode` - Request code (should be `PICK_IMAGE`)
- `resultCode` - Result code from the picker
- `data` - Intent containing selected image URI

**Workflow**:
1. Load PyTorch model
2. Get selected image URI
3. Display image in ImageView
4. Preprocess image (resize, normalize)
5. Convert to tensor
6. Run inference
7. Find top prediction
8. Display class name

---

## 🖼️ Image Processing

### Preprocessing Pipeline

#### Step 1: Bitmap Loading

```java
Bitmap bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
```

#### Step 2: Resizing

```java
Bitmap resized = Bitmap.createScaledBitmap(bitmap, 320, 320, false);
```

**Parameters**:
- `bitmap` - Source bitmap
- `width` - Target width (320)
- `height` - Target height (320)
- `filter` - Whether to filter (false for faster processing)

#### Step 3: Tensor Conversion

```java
Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
    bitmap,
    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
    TensorImageUtils.TORCHVISION_NORM_STD_RGB
);
```

**Normalization Values**:
- Mean RGB: [0.485, 0.456, 0.406]
- Std RGB: [0.229, 0.224, 0.225]

---

## 🧠 Model Operations

### Loading the Model

```java
Module module = Module.load(assetFilePath(context, "model.pt"));
```

**Description**: Loads a TorchScript model from the assets folder.

**Parameters**:
- `modelPath` - Absolute path to the model file

**Returns**: `Module` instance ready for inference

**Best Practices**:
- Load model once and reuse
- Load asynchronously to avoid blocking UI
- Handle loading errors gracefully

---

### Running Inference

```java
Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
```

**Description**: Performs forward pass through the model.

**Parameters**:
- `inputTensor` - Preprocessed input tensor (1, 3, 320, 320)

**Returns**: Output tensor with class scores (1, 1000)

**Performance**:
- Average time: 50-100ms
- Device dependent
- No GPU acceleration on most devices

---

### Processing Results

```java
float[] scores = outputTensor.getDataAsFloatArray();

float maxScore = -Float.MAX_VALUE;
int maxScoreIdx = -1;

for (int i = 0; i < scores.length; i++) {
    if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
    }
}

String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
```

**Description**: Extracts the top prediction from model output.

**Output Format**:
- Array of 1000 float values
- Values are logits (pre-softmax)
- Higher value = higher confidence

**Improvements**:
- Add softmax for probability scores
- Return top-K predictions
- Add confidence threshold

---

## 📊 Data Structures

### ImageNet Classes

```java
public class ImageNetClasses {
    public static final String[] IMAGENET_CLASSES = {
        "tench",
        "goldfish",
        "great white shark",
        // ... 997 more classes
    };
}
```

**Description**: Array containing all 1000 ImageNet class labels.

**Usage**:
```java
String label = ImageNetClasses.IMAGENET_CLASSES[predictedIndex];
```

---

## 🎯 Tensor Specifications

### Input Tensor

```
Shape: [1, 3, 320, 320]
Type: Float32
Format: NCHW (Batch, Channels, Height, Width)
Range: [-2.64, 2.64] (after normalization)
```

### Output Tensor

```
Shape: [1, 1000]
Type: Float32
Format: Logits (pre-softmax scores)
Range: Unbounded (typically -20 to 20)
```

---

## 🔄 Advanced Usage

### Batch Processing

```java
// Future enhancement
List<Bitmap> images = loadImages();
Tensor batchTensor = createBatchTensor(images);
Tensor output = module.forward(IValue.from(batchTensor)).toTensor();
```

### Top-K Predictions

```java
public List<Prediction> getTopK(float[] scores, int k) {
    // Sort scores and return top K predictions
    List<Prediction> results = new ArrayList<>();
    // Implementation here
    return results;
}
```

### Confidence Scores

```java
public float[] softmax(float[] logits) {
    float[] probabilities = new float[logits.length];
    float sum = 0.0f;

    for (float logit : logits) {
        sum += Math.exp(logit);
    }

    for (int i = 0; i < logits.length; i++) {
        probabilities[i] = (float) (Math.exp(logits[i]) / sum);
    }

    return probabilities;
}
```

---

## ⚠️ Error Handling

### Common Errors

#### Model Loading Failure

```java
try {
    module = Module.load(assetFilePath(this, "model.pt"));
} catch (IOException e) {
    Log.e("ModelError", "Failed to load model", e);
    Toast.makeText(this, "Model loading failed", Toast.LENGTH_SHORT).show();
}
```

#### Out of Memory

```java
try {
    // Inference code
} catch (OutOfMemoryError e) {
    System.gc();
    Log.e("MemoryError", "Out of memory during inference", e);
}
```

#### Invalid Image Format

```java
if (bitmap == null) {
    Toast.makeText(this, "Invalid image format", Toast.LENGTH_SHORT).show();
    return;
}
```

---

## 🧪 Testing

### Unit Test Example

```java
@Test
public void testImagePreprocessing() {
    Bitmap testImage = createTestBitmap(320, 320);
    Tensor tensor = preprocessImage(testImage);

    assertEquals(1, tensor.shape()[0]); // Batch size
    assertEquals(3, tensor.shape()[1]); // Channels
    assertEquals(320, tensor.shape()[2]); // Height
    assertEquals(320, tensor.shape()[3]); // Width
}
```

### Integration Test Example

```java
@Test
public void testModelInference() {
    Module module = loadModel();
    Bitmap testImage = loadTestImage();
    Tensor input = preprocessImage(testImage);

    Tensor output = module.forward(IValue.from(input)).toTensor();

    assertNotNull(output);
    assertEquals(1000, output.shape()[1]);
}
```

---

## 📖 Code Examples

### Complete Classification Example

```java
public String classifyImage(Bitmap bitmap) {
    try {
        // 1. Load model (do this once)
        if (module == null) {
            module = Module.load(assetFilePath(context, "model.pt"));
        }

        // 2. Preprocess
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 320, 320, false);
        Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );

        // 3. Inference
        Tensor output = module.forward(IValue.from(input)).toTensor();

        // 4. Get result
        float[] scores = output.getDataAsFloatArray();
        int maxIdx = findMaxIndex(scores);

        return ImageNetClasses.IMAGENET_CLASSES[maxIdx];

    } catch (Exception e) {
        Log.e("Classification", "Error during inference", e);
        return "Error";
    }
}
```

---

## 🔗 Related Documentation

- [Architecture Documentation](ARCHITECTURE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md)

---

**Last Updated**: 2024-2025

**API Version**: 1.0.0
