import os
import subprocess
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
import requests

OUTPUT_DIR = "./output_models"
RAW_DIR = "./raw_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

print("=== 1. Downloading PP-OCRv6 Models & Dictionary ===")

# Download Detection Medium
det_raw = hf_hub_download(
    repo_id="PaddlePaddle/PP-OCRv6_medium_det_onnx",
    filename="inference.onnx",
    local_dir=f"{RAW_DIR}/det",
)

# Download Recognition Medium (Unified 50 Languages)
rec_raw = hf_hub_download(
    repo_id="PaddlePaddle/PP-OCRv6_medium_rec_onnx",
    filename="inference.onnx",
    local_dir=f"{RAW_DIR}/rec",
)

# Download PP-OCRv6 Dictionary
dict_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv6_dict.txt"
dict_dest = os.path.join(OUTPUT_DIR, "ppocrv6_dict.txt")
print(f"Downloading dictionary from {dict_url}...")
resp = requests.get(dict_url)
resp.raise_for_status()
with open(dict_dest, "wb") as f:
    f.write(resp.content)
print(f"Saved dictionary to: {dict_dest}")

print("\n=== 2. Running onnxslim Graph Simplification ===")
det_slim = os.path.join(OUTPUT_DIR, "det_v6_medium.slim.onnx")
rec_slim = os.path.join(OUTPUT_DIR, "rec_v6_medium.slim.onnx")

subprocess.run(["onnxslim", det_raw, det_slim], check=True)
subprocess.run(["onnxslim", rec_raw, rec_slim], check=True)

print(f"Simplified Detection: {det_slim} ({os.path.getsize(det_slim) / 1e6:.2f} MB)")
print(f"Simplified Recognition: {rec_slim} ({os.path.getsize(rec_slim) / 1e6:.2f} MB)")

print("\n=== 3. Running Dynamic INT8 Quantization on Recognition Model ===")
rec_int8 = os.path.join(OUTPUT_DIR, "rec_v6_medium.slim.int8.onnx")

quantize_dynamic(
    model_input=rec_slim,
    model_output=rec_int8,
    weight_type=QuantType.QUInt8,
    per_channel=True,
)

print(f"INT8 Quantized Recognition: {rec_int8} ({os.path.getsize(rec_int8) / 1e6:.2f} MB)")

print("\n=== Final Deployment Assets in ./output_models/ ===")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f" - {fname} ({os.path.getsize(fpath) / 1e6:.2f} MB)")
