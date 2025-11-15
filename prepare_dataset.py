import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==== CONFIG ====
AUDIO_BASE = "commonvoice"
MEL_DIR = "processed_data_mfcc_augmented"  # Changed to augmented folder
MIN_SAMPLES = 23000  # minimum samples per accent
MAX_SAMPLES = 24000  # cap per accent to balance classes
# ================

# First pass: determine valid accents from train set
valid_accents = None

# Process the three splits
for split in ["train", "dev", "test"]:
    CSV_PATH = f"commonvoice/cv-valid-{split}.csv"
    OUTPUT_FILE = f"labeled_dataset_{split}.csv"

    print(f"\n=== Processing {split} split===")

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
        base_name = os.path.basename(rel_path).replace(".mp3", "")
        
        # For train: check all 5 augmented versions
        if split == "train":
            for aug_idx in range(6):
                mel_path = os.path.join(MEL_DIR, f"{base_name}_aug{aug_idx}.npy")
                if os.path.exists(mel_path):
                    valid_files.append(mel_path)
                    labels.append(accent)
        else:
            # For dev/test: only original (aug0)
            mel_path = os.path.join(MEL_DIR, f"{base_name}_aug0.npy")
            if os.path.exists(mel_path):
                valid_files.append(mel_path)
                labels.append(accent)

    # Create dataframe
    out_df = pd.DataFrame({"spectrogram": valid_files, "accent": labels})
    
    # ==== CLASS BALANCING ====
    # Count samples per accent
    accent_counts = out_df['accent'].value_counts()
    print(f"\nOriginal accent distribution:\n{accent_counts}")
    
    if split == "train":
        # Determine valid accents from train set only
        valid_accents = accent_counts[accent_counts >= MIN_SAMPLES].index
        print(f"\nAccents with >= {MIN_SAMPLES} samples in train: {list(valid_accents)}")
        
        # Filter and cap train set
        out_df = out_df[out_df['accent'].isin(valid_accents)]
        
        # Cap: sample MAX_SAMPLES from each accent (after augmentation)
        # Note: With 5x augmentation, this is effectively MAX_SAMPLES total per accent
        balanced_dfs = []
        for accent in valid_accents:
            accent_df = out_df[out_df['accent'] == accent]
            if len(accent_df) > MAX_SAMPLES:
                accent_df = accent_df.sample(n=MAX_SAMPLES, random_state=42)
            balanced_dfs.append(accent_df)
        
        out_df = pd.concat(balanced_dfs, ignore_index=True)
        out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    else:
        # For dev/test: use the same accents determined from train
        out_df = out_df[out_df['accent'].isin(valid_accents)]
        out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    
    print(f"\nFinal accent distribution:\n{out_df['accent'].value_counts()}")
    # =========================
    
    # Save to labeled CSV
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(out_df)} labeled samples to {OUTPUT_FILE}")