# Baidu PaddleOCR PP-OCRv6 Medium - Mobile ONNX Models

Optimized ONNX Runtime deployment assets for Baidu's latest **PP-OCRv6 Medium** models.

## Models Included

1. **Detection Medium**: `det_v6_medium.slim.onnx`
   - Optimized with `onnxslim` to prune dead graph nodes, fold constant operations, and clean up reparameterized PPLCNetV4 blocks.
2. **Recognition Medium (50 Languages)**: `rec_v6_medium.slim.int8.onnx`
   - Cleaned with `onnxslim` and quantized via dynamic INT8 (`onnxruntime.quantization`) with `QuantType.QUInt8` and per-channel calibration for maximum mobile performance and minimal footprint.
3. **Character Dictionary**: `ppocrv6_dict.txt`
   - Official 50-language dictionary from PaddleOCR.

## Pipeline & Optimization

The models are processed using `uv` and Python:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install huggingface_hub onnx onnxruntime onnxslim requests
uv run python optimize_models.py
```

### Optimizations Applied
- **Graph Simplification (`onnxslim`)**:
  - Eliminates redundant constants and identities.
  - Folds batch-normalization and constant-folding layers.
  - Removes unused branches and simplifies reparameterized convolutional blocks.
- **Dynamic Quantization (`onnxruntime.quantization`)**:
  - Dynamically quantizes feedforward weights to 8-bit unsigned integers (`QUInt8`).
  - Per-channel weight quantization for high fidelity and accuracy retention.
