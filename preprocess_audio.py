import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed  # <-- ADDED IMPORT

# ==== CONFIG ====
AUDIO_BASE = "commonvoice"
OUTPUT_DIR = "processed_data_mfcc_augmented"
SAMPLE_RATE = 16000
N_MFCC = 20
# ================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_mfcc(y, sr, n_mfcc=N_MFCC):
    """Extract MFCC features from audio"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    return features

def augment_and_extract(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Load audio, apply augmentation, extract features"""
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)
    
    augmented_features = []
    
    # 1. Original
    augmented_features.append(extract_mfcc(y, sr, n_mfcc))
    
    # 2. Time stretch (faster)
    y_fast = librosa.effects.time_stretch(y, rate=1.1)
    augmented_features.append(extract_mfcc(y_fast, sr, n_mfcc))
    
    # 3. Time stretch (slower)
    y_slow = librosa.effects.time_stretch(y, rate=0.9)
    augmented_features.append(extract_mfcc(y_slow, sr, n_mfcc))
    
    # 4. Pitch shift (up)
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    augmented_features.append(extract_mfcc(y_pitch_up, sr, n_mfcc))

    # 5. Pitch shift (down)
    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    augmented_features.append(extract_mfcc(y_pitch_down, sr, n_mfcc))
    
    # 6. Add noise
    noise = np.random.randn(len(y)) * 0.005
    y_noise = y + noise
    augmented_features.append(extract_mfcc(y_noise, sr, n_mfcc))
    
    return augmented_features

# <-- ADDED HELPER FUNCTION FOR PARALLELISM -->
def process_file(file_path, split):
    """
    Processes a single audio file.
    This function contains the logic from the original for-loop.
    """
    try:
        base_name = os.path.basename(file_path).replace(".mp3", "")
        
        # Only augment training data
        if split == "train":
            augmented_list = augment_and_extract(file_path)
            # Save each augmented version
            for i, features in enumerate(augmented_list):
                save_path = os.path.join(OUTPUT_DIR, f"{base_name}_aug{i}.npy")
                np.save(save_path, features)
        else:
            # Dev/test: just original
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            y = librosa.util.normalize(y)
            features = extract_mfcc(y, sr, N_MFCC)
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_aug0.npy")
            np.save(save_path, features)
            
    except Exception as e:
        # It's good practice to print which file failed
        print(f"Error processing {file_path}: {e}")
# <-- END OF ADDED FUNCTION -->


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        CSV_PATH = f"commonvoice/cv-valid-{split}.csv"
        print(f"\n=== Processing {split} split ===")
        
        df = pd.read_csv(CSV_PATH)
        
        file_list = []
        for rel_path in df["filename"].dropna():
            full_path = os.path.join(AUDIO_BASE, rel_path)
            if os.path.exists(full_path):
                file_list.append(full_path)
        
        print(f"Found {len(file_list)} valid audio files. Processing with augmentation...")
        
        # <-- REPLACED LOOP WITH PARALLEL CALL -->
        # We wrap file_list with tqdm() to get the progress bar
        n_jobs = 4 #mac friendly script for lucy
        # n_jobs = 24 #Williams machine
        Parallel(n_jobs)(
            delayed(process_file)(fp, split) for fp in tqdm(file_list)
        )
        # <-- END OF REPLACEMENT -->
            
        print(f"Done! {split} features saved in: {OUTPUT_DIR}")