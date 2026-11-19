# Baidu PaddleOCR PP-OCRv6 Mobile ONNX Models (Medium & Small)

Optimized ONNX Runtime deployment assets for Baidu's latest **PP-OCRv6** models, organized into **Medium** and **Small** tiers for edge and mobile deployment.

---

## Directory Structure & Assets

All deployment assets are structured by tier in `./output_models/`:

```text
output_models/
├── medium/
│   ├── det_v6_medium.slim.onnx       (62.06 MB) - FP32 Detection (onnxslim)
│   ├── det_v6_medium.slim.int8.onnx  (15.87 MB) - Dynamic INT8 Detection
│   ├── rec_v6_medium.slim.onnx       (76.55 MB) - FP32 Recognition (onnxslim)
│   ├── rec_v6_medium.slim.int8.onnx  (19.57 MB) - Dynamic INT8 Recognition (50 Languages)
│   └── ppocrv6_dict.txt              ( 0.07 MB) - 50-language dictionary
├── small/
│   ├── det_v6_small.slim.onnx        ( 9.92 MB) - FP32 Detection (onnxslim)
│   ├── det_v6_small.slim.int8.onnx   ( 2.68 MB) - Dynamic INT8 Detection
│   ├── rec_v6_small.slim.onnx        (21.15 MB) - FP32 Recognition (onnxslim)
│   ├── rec_v6_small.slim.int8.onnx   ( 5.65 MB) - Dynamic INT8 Recognition (50 Languages)
│   └── ppocrv6_dict.txt              ( 0.07 MB) - 50-language dictionary
└── ppocrv6_dict.txt                  ( 0.07 MB) - Shared root dictionary
```

---

## Model Benchmarks & Comparison

### 1. Medium Suite (`./output_models/medium/`)
| Asset Name | Precision | Original Size | Optimized Size | Size Reduction | Target Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `det_v6_medium.slim.onnx` | FP32 | 62.03 MB | **62.06 MB** | — | High-accuracy detection (PPLCNetV4 backbone) |
| `det_v6_medium.slim.int8.onnx` | INT8 | 62.03 MB | **15.87 MB** | **74.4%** | Fast mobile detection with high recall |
| `rec_v6_medium.slim.onnx` | FP32 | 76.56 MB | **76.55 MB** | — | Reference FP32 recognition (50 languages) |
| `rec_v6_medium.slim.int8.onnx` | INT8 | 76.56 MB | **19.57 MB** | **74.4%** | **Recommended mobile default**: High accuracy under 20 MB |

### 2. Small Suite (`./output_models/small/`)
| Asset Name | Precision | Original Size | Optimized Size | Size Reduction | Target Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `det_v6_small.slim.onnx` | FP32 | 9.88 MB | **9.92 MB** | — | Lightweight detection |
| `det_v6_small.slim.int8.onnx` | INT8 | 9.88 MB | **2.68 MB** | **72.9%** | **Ultra-low latency mobile detector** (< 3 MB) |
| `rec_v6_small.slim.onnx` | FP32 | 21.16 MB | **21.15 MB** | — | Lightweight FP32 recognition |
| `rec_v6_small.slim.int8.onnx` | INT8 | 21.16 MB | **5.65 MB** | **73.3%** | **Ultra-lightweight recognition** (< 6 MB, 50 languages) |

> **Performance Note**: `rec_v6_medium.slim.int8.onnx` (**19.57 MB**) has a smaller file size than `rec_v6_small.slim.onnx` (**21.15 MB**) while delivering significantly higher recognition accuracy on degraded images, curved text, and rare multi-lingual fonts.

---

## Usage Instructions

### 1. Python (ONNX Runtime) Quickstart

Install dependencies:
```bash
pip install onnxruntime numpy opencv-python Pillow
```

#### Running Text Recognition:
```python
import numpy as np
import onnxruntime as ort
from PIL import Image

# 1. Load dictionary
with open("output_models/medium/ppocrv6_dict.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip("\r\n") for line in f.readlines()]
# In PP-OCR CTC decoding: index 0 is blank, followed by vocab tokens, then space
character_dict = ["blank"] + vocab + [" "]

# 2. Initialize ONNX Runtime Session (optimized for mobile CPU)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "output_models/medium/rec_v6_medium.slim.int8.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)

# 3. Preprocess image crop (fixed height=48, dynamic width W)
img = Image.open("sample_crop.jpg").convert("RGB")
w, h = img.size
ratio = w / float(h)
target_w = max(16, int(round(48 * ratio / 32) * 32))  # aligned to 32
img_resized = img.resize((target_w, 48), Image.Resampling.BILINEAR)

# Normalize: (pixel / 255.0 - 0.5) / 0.5 -> [-1.0, 1.0]
norm_img = (np.array(img_resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
input_tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, :, :, :]  # [1, 3, 48, W]

# 4. Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
preds = session.run([output_name], {input_name: input_tensor})[0]  # [1, T, 18710]

# 5. CTC Greedy Decoding
pred_indices = np.argmax(preds[0], axis=-1)
pred_probs = np.max(preds[0], axis=-1)

recognized_chars = []
confidence_scores = []
prev_idx = 0
for idx, prob in zip(pred_indices, pred_probs):
    if idx != 0 and idx != prev_idx:  # ignore blank and consecutive duplicates
        if idx < len(character_dict):
            recognized_chars.append(character_dict[idx])
            confidence_scores.append(prob)
    prev_idx = idx

text = "".join(recognized_chars)
avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
print(f"Recognized Text: '{text}' (Confidence: {avg_confidence:.2f})")
```

#### Running Text Detection:
```python
import cv2
import numpy as np
import onnxruntime as ort

# Load Detection Session
det_session = ort.InferenceSession("output_models/medium/det_v6_medium.slim.onnx")

# Preprocessing: resize to multiples of 32 and ImageNet normalize
img = cv2.imread("document.jpg")
h, w, _ = img.shape
scale_h = max(32, int(round(h / 32) * 32))
scale_w = max(32, int(round(w / 32) * 32))
resized = cv2.resize(img, (scale_w, scale_h))

# Normalize: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
norm = resized.astype(np.float32) / 255.0
norm = (norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
input_tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

# Inference output is probability map [1, 1, H, W]
pred = det_session.run(None, {det_session.get_inputs()[0].name: input_tensor})[0]
prob_map = pred[0, 0, :, :]
bitmap = (prob_map > 0.3).astype(np.uint8)
```

---

### 2. Mobile Deployment Architecture

#### Recommended Pairings:
1. **Balanced Production (Default)**:
   - Detection: `output_models/medium/det_v6_medium.slim.onnx` (or `det_v6_medium.slim.int8.onnx`)
   - Recognition: `output_models/medium/rec_v6_medium.slim.int8.onnx`
   - *Best for documents, invoices, receipts, and natural scene text.*
2. **High-Speed / Viewfinder Scan**:
   - Detection: `output_models/small/det_v6_small.slim.int8.onnx`
   - Recognition: `output_models/small/rec_v6_small.slim.int8.onnx`
   - *Best for low-end Android devices, real-time camera viewfinder detection (20-30 FPS).*

#### Android (Kotlin / Java)
Include the ONNX Runtime Android package:
```groovy
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'
```
Enable the XNNPACK / NNAPI execution provider for hardware acceleration:
```kotlin
val env = OrtEnvironment.getEnvironment()
val sessionOptions = OrtSession.SessionOptions().apply {
    setIntraOpNumThreads(4)
    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    addXnnpack(mapOf("intra_op_num_threads" to "4"))
}
val session = env.createSession(modelBytes, sessionOptions)
```

#### iOS (Swift)
Include the CocoaPod:
```ruby
pod 'onnxruntime-c' # or onnxruntime-objc
```
Set session options with CoreML or CPU execution provider:
```swift
let options = try ORTSessionOptions()
try options.setIntraOpNumThreads(4)
let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
```

#### Flutter (`flutter_onnxruntime`)
Use the modern, actively maintained [`flutter_onnxruntime`](https://pub.dev/packages/flutter_onnxruntime) package:

```yaml
dependencies:
  flutter_onnxruntime: ^1.9.0
```

```dart
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

Future<void> runOcrInference(List<double> normalizedPixels, int targetW) async {
  final ort = OnnxRuntime();

  // Create session from Flutter asset or local device file
  final session = await ort.createSessionFromAsset(
    'assets/models/rec_v6_medium.slim.int8.onnx',
  );

  // Prepare input tensor [1, 3, 48, targetW]
  final inputName = session.inputNames.first;
  final inputTensor = await OrtValue.fromList(
    normalizedPixels,
    [1, 3, 48, targetW],
  );

  try {
    // Run inference
    final outputs = await session.run({inputName: inputTensor});
    final outputName = session.outputNames.first;
    final outputTensor = outputs[outputName];

    if (outputTensor != null) {
      final results = await outputTensor.asList();
      // Decode CTC tokens using ppocrv6_dict.txt
    }

    // Clean up native output tensors
    for (final tensor in outputs.values) {
      tensor.dispose();
    }
  } finally {
    // Free native memory
    inputTensor.dispose();
    await session.close();
  }
}
```

---

## Optimization Pipeline Reproduction

To reproduce or re-run the download and optimization pipeline using `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install huggingface_hub onnx onnxruntime onnxslim requests
uv run python optimize_models.py
```
