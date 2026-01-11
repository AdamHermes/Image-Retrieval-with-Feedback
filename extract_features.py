import os
import json
import numpy as np
from tqdm import tqdm
from clip_encoder import CLIPEncoder

# =========================
# Config
# =========================
IMAGE_ROOT = "data/img"
OUT_DIR = "features"
IMAGE_SUFFIX = ["_front.jpg","_back.jpg","_flat.jpg"]   # change to ".jpg" if you want all views

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Init encoder
# =========================
encoder = CLIPEncoder()

image_paths = []
embeddings = []

# =========================
# Collect image paths
# =========================
for root, _, files in os.walk(IMAGE_ROOT):
    for file in files:
        for suffix in IMAGE_SUFFIX:
            if file.lower().endswith(suffix):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

print(f"[INFO] Found {len(image_paths)} images")

assert len(image_paths) > 0, "❌ No images found — check IMAGE_ROOT or filename pattern"

# =========================
# Extract features
# =========================
for path in tqdm(image_paths, desc="Extracting CLIP features"):
    emb = encoder.encode_image(path)      # (1, D) or (D,)
    emb = emb.detach().cpu().numpy()      # safe for torch tensors
    embeddings.append(emb)

# =========================
# Stack + save
# =========================
embeddings = np.vstack(embeddings)

np.save(os.path.join(OUT_DIR, "image_embeddings.npy"), embeddings)

with open(os.path.join(OUT_DIR, "image_paths.json"), "w") as f:
    json.dump(image_paths, f, indent=2)

print("[INFO] Feature extraction complete.")
print(f"[INFO] Saved embeddings: {embeddings.shape}")
print(f"[INFO] Saved paths to: {OUT_DIR}/image_paths.json")
