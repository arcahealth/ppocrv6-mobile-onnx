# Baidu PaddleOCR PP-OCRv6 Mobile ONNX Models (Medium & Small)

Optimized ONNX Runtime deployment assets for Baidu's latest **PP-OCRv6** models, organized into **Medium** and **Small** directories.

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

## Model Benchmarks

### 1. Medium Suite (`./output_models/medium/`)
| Asset Name | Precision | Original Size | Optimized Size | Size Reduction | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `det_v6_medium.slim.onnx` | FP32 | 62.03 MB | **62.06 MB** | — | Cleaned via `onnxslim` (PPLCNetV4 fusion) |
| `det_v6_medium.slim.int8.onnx` | INT8 | 62.03 MB | **15.87 MB** | **74.4%** | Dynamic quantization (`QUInt8`, per-channel) |
| `rec_v6_medium.slim.onnx` | FP32 | 76.56 MB | **76.55 MB** | — | Cleaned via `onnxslim` (143 identity ops pruned) |
| `rec_v6_medium.slim.int8.onnx` | INT8 | 76.56 MB | **19.57 MB** | **74.4%** | Dynamic quantization (`QUInt8`, per-channel) |

### 2. Small Suite (`./output_models/small/`)
| Asset Name | Precision | Original Size | Optimized Size | Size Reduction | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `det_v6_small.slim.onnx` | FP32 | 9.88 MB | **9.92 MB** | — | Cleaned via `onnxslim` |
| `det_v6_small.slim.int8.onnx` | INT8 | 9.88 MB | **2.68 MB** | **72.9%** | Ultra-lightweight mobile detector |
| `rec_v6_small.slim.onnx` | FP32 | 21.16 MB | **21.15 MB** | — | Cleaned via `onnxslim` (135 identity ops pruned) |
| `rec_v6_small.slim.int8.onnx` | INT8 | 21.16 MB | **5.65 MB** | **73.3%** | 50-language recognition under 6 MB |

---

## Pipeline Execution

The pipeline is managed via `uv`:

```bash
# 1. Create and activate venv
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install huggingface_hub onnx onnxruntime onnxslim requests

# 3. Run optimization pipeline
uv run python optimize_models.py
```

### Optimizations Applied
- **Graph Simplification (`onnxslim`)**:
  - Eliminates redundant constants, identity layers, and shape manipulation ops.
  - Folds batch-normalization layers directly into convolution kernels.
  - Resolves reparameterized convolutional blocks in PPLCNetV4 backbones.
- **Dynamic Quantization (`onnxruntime.quantization`)**:
  - Dynamically quantizes feedforward weights to 8-bit unsigned integers (`QuantType.QUInt8`).
  - Utilizes per-channel weight calibration for high fidelity and accuracy retention on mobile engines (XNNPACK, NNAPI, CoreML).
