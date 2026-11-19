# Baidu PaddleOCR PP-OCRv6 Mobile ONNX Models (Medium & Small)

Optimized ONNX Runtime deployment assets for Baidu's latest **PP-OCRv6** models, supporting both **Medium** and ultra-lightweight **Small** tiers.

## Model Benchmarks & Assets

All assets are generated into `./output_models/`:

### 1. Medium Models
| Asset Name | Precision | Original Size | Optimized Size | Size Reduction | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `det_v6_medium.slim.onnx` | FP32 | 62.03 MB | **62.06 MB** | — | Cleaned via `onnxslim` (PPLCNetV4 fusion) |
| `det_v6_medium.slim.int8.onnx` | INT8 | 62.03 MB | **15.87 MB** | **74.4%** | Dynamic quantization (`QUInt8`, per-channel) |
| `rec_v6_medium.slim.onnx` | FP32 | 76.56 MB | **76.55 MB** | — | Cleaned via `onnxslim` (143 identity ops pruned) |
| `rec_v6_medium.slim.int8.onnx` | INT8 | 76.56 MB | **19.57 MB** | **74.4%** | Dynamic quantization (`QUInt8`, per-channel) |

### 2. Small Models
| Asset Name | Precision | Original Size | Optimized Size | Size Reduction | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `det_v6_small.slim.onnx` | FP32 | 9.88 MB | **9.92 MB** | — | Cleaned via `onnxslim` |
| `det_v6_small.slim.int8.onnx` | INT8 | 9.88 MB | **2.68 MB** | **72.9%** | Ultra-lightweight mobile detector |
| `rec_v6_small.slim.onnx` | FP32 | 21.16 MB | **21.15 MB** | — | Cleaned via `onnxslim` (135 identity ops pruned) |
| `rec_v6_small.slim.int8.onnx` | INT8 | 21.16 MB | **5.65 MB** | **73.3%** | 50-language recognition under 6 MB |

### 3. Dictionary
- `ppocrv6_dict.txt`: Official PaddleOCR multi-lingual dictionary covering 50 languages (18,708 character tokens).

---

## Pipeline Execution

The repository uses `uv` for reproducible environment management:

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
  - Prunes dead graph nodes, identities, and redundant squeeze/unsqueeze operations.
  - Folds batch-normalization layers directly into convolution kernels.
  - Unifies multi-branch reparameterized blocks in PPLCNetV4 backbones.
- **Dynamic Quantization (`onnxruntime.quantization`)**:
  - Dynamically quantizes feedforward weights to 8-bit unsigned integers (`QuantType.QUInt8`).
  - Utilizes per-channel weight calibration for maximum accuracy retention.
