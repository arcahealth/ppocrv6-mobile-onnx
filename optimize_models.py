import os
import shutil
import subprocess
import sys
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
import requests

OUTPUT_DIR = "./output_models"
RAW_DIR = "./raw_models"
MEDIUM_DIR = os.path.join(OUTPUT_DIR, "medium")
SMALL_DIR = os.path.join(OUTPUT_DIR, "small")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MEDIUM_DIR, exist_ok=True)
os.makedirs(SMALL_DIR, exist_ok=True)

print("=== 1. Downloading PP-OCRv6 Models & Dictionary ===")

# Model definitions: (name, repo_id, raw_subdir, out_subdir)
models_to_download = [
    # Medium Models
    ("det_v6_medium", "PaddlePaddle/PP-OCRv6_medium_det_onnx", "det", MEDIUM_DIR),
    ("rec_v6_medium", "PaddlePaddle/PP-OCRv6_medium_rec_onnx", "rec", MEDIUM_DIR),
    # Small Models
    ("det_v6_small", "PaddlePaddle/PP-OCRv6_small_det_onnx", "det_small", SMALL_DIR),
    ("rec_v6_small", "PaddlePaddle/PP-OCRv6_small_rec_onnx", "rec_small", SMALL_DIR),
]

raw_paths = {}
for name, repo_id, subdir, _ in models_to_download:
    dest_dir = os.path.join(RAW_DIR, subdir)
    dest_path = os.path.join(dest_dir, "inference.onnx")
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"Model {name} already exists at {dest_path}, skipping download.")
        raw_paths[name] = dest_path
    else:
        print(f"Downloading {name} from {repo_id}...")
        raw_paths[name] = hf_hub_download(
            repo_id=repo_id,
            filename="inference.onnx",
            local_dir=dest_dir,
        )

# Download PP-OCRv6 Dictionary (if not present)
dict_dest = os.path.join(OUTPUT_DIR, "ppocrv6_dict.txt")
if not os.path.exists(dict_dest):
    dict_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv6_dict.txt"
    print(f"Downloading dictionary from {dict_url}...")
    resp = requests.get(dict_url)
    resp.raise_for_status()
    with open(dict_dest, "wb") as f:
        f.write(resp.content)
    print(f"Saved dictionary to: {dict_dest}")
else:
    print(f"Dictionary already present at: {dict_dest}")

# Copy dictionary to medium/ and small/ subdirectories
shutil.copyfile(dict_dest, os.path.join(MEDIUM_DIR, "ppocrv6_dict.txt"))
shutil.copyfile(dict_dest, os.path.join(SMALL_DIR, "ppocrv6_dict.txt"))

print("\n=== 2. Running onnxslim Graph Simplification ===")
slim_paths = {}
for name, _, _, out_dir in models_to_download:
    slim_dest = os.path.join(out_dir, f"{name}.slim.onnx")
    slim_paths[name] = slim_dest
    print(f"Slimming {name} -> {slim_dest}...")
    subprocess.run([sys.executable, "-m", "onnxslim", raw_paths[name], slim_dest], check=True)
    print(f"Simplified {name}: {slim_dest} ({os.path.getsize(slim_dest) / 1e6:.2f} MB)")

print("\n=== 3. Running Dynamic INT8 Quantization ===")
for name, _, _, out_dir in models_to_download:
    int8_dest = os.path.join(out_dir, f"{name}.slim.int8.onnx")
    print(f"Quantizing {name} -> {int8_dest}...")
    quantize_dynamic(
        model_input=slim_paths[name],
        model_output=int8_dest,
        weight_type=QuantType.QUInt8,
        per_channel=True,
    )
    print(f"INT8 Quantized {name}: {int8_dest} ({os.path.getsize(int8_dest) / 1e6:.2f} MB)")

print("\n=== Final Deployment Assets in ./output_models/ ===")
for root, _, files in os.walk(OUTPUT_DIR):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        rel_path = os.path.relpath(fpath, OUTPUT_DIR)
        print(f" - {rel_path:<35} ({os.path.getsize(fpath) / 1e6:.2f} MB)")
