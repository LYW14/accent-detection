import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==== CONFIG ====
CSV_PATH = "commonvoice/cv-other-train.csv"
AUDIO_BASE = "commonvoice"
MEL_DIR = "processed_data"  # where your .npy spectrograms are saved
OUTPUT_FILE = "labeled_dataset.csv"
# ================

# Load CSV
df = pd.read_csv(CSV_PATH)

# Keep only rows with accent labels
df = df.dropna(subset=["accent"])
df = df[df["accent"].str.strip() != ""]

# Keep only files that exist in both CSV and processed_data
valid_files = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    rel_path = row["filename"]
    accent = row["accent"].strip().lower()
    mel_name = os.path.basename(rel_path).replace(".mp3", ".npy")
    mel_path = os.path.join(MEL_DIR, mel_name)
    if os.path.exists(mel_path):
        valid_files.append(mel_path)
        labels.append(accent)

# Save to labeled CSV
out_df = pd.DataFrame({"spectrogram": valid_files, "accent": labels})
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(out_df)} labeled samples to {OUTPUT_FILE}")
