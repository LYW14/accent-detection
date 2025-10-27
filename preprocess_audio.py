import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#ACTION ITEM: Change this to dataset path !!!!!
DATA_DIR = "path/to/commonvoice/audio"
OUTPUT_DIR = "preprocessed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_audio(file_path, sr=16000, n_mels=128):
    """Load, normalize, and create mel spectrogram"""
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def visualize_spectrogram(S_dB):
    """Show the mel spectrogram"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=16000, hop_length=512, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]

    for file in tqdm(all_files):
        file_path = os.path.join(DATA_DIR, file)
        S_dB = preprocess_audio(file_path)
        np.save(os.path.join(OUTPUT_DIR, file.replace(".wav", ".npy")), S_dB)