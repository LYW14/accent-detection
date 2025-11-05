import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display

# ==== CONFIG ====
AUDIO_BASE = "commonvoice"
OUTPUT_DIR = "processed_data"
SAMPLE_RATE = 16000
N_MELS = 128
# ================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_audio(file_path, sr=SAMPLE_RATE, n_mels=N_MELS):
    """Load, normalize, and create mel spectrogram"""
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

if __name__ == "__main__":
    for split in ["train", "dev", "test"]: #processing each split stored in different folders
        CSV_PATH = f"commonvoice/cv-valid-{split}.csv"

        print(f"\n=== Processing {split} split ===")

        df = pd.read_csv(CSV_PATH)

        # Filter for files that actually exist
        file_list = []
        for rel_path in df["filename"].dropna():
            full_path = os.path.join(AUDIO_BASE, rel_path)
            if os.path.exists(full_path):
                file_list.append(full_path)

        print(f"Found {len(file_list)} valid audio files. Processing...")

        for file_path in tqdm(file_list):  # limit to 100 for now
            try:
                S_dB = preprocess_audio(file_path)
                np.save(os.path.join(OUTPUT_DIR, os.path.basename(file_path).replace(".mp3", ".npy")), S_dB)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print(f"Done! {split} spectrograms saved in:", OUTPUT_DIR)