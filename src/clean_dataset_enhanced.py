import os
import numpy as np
import librosa
from scipy.io import wavfile
from tqdm import tqdm

DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset_cleaned_final"
SAMPLE_RATE = 16000

def clean_and_save(input_path, output_path):
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    except:
        return False
    if len(audio) == 0:
        return False

    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

    if len(audio_trimmed) < int(0.2 * SAMPLE_RATE):
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, SAMPLE_RATE, (audio_trimmed * 32767).astype(np.int16))
    return True

wav_files = []
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(".wav"):
            full = os.path.join(root, file)
            rel = os.path.relpath(full, DATASET_DIR)
            out = os.path.join(OUTPUT_DIR, rel)
            wav_files.append((full, out))

print(f"Found {len(wav_files)} files")
success = 0
for inp, out in tqdm(wav_files, desc="Trimming"):
    if clean_and_save(inp, out):
        success += 1
    else:
        print(f"Skipped (too short?): {inp}")

print(f"{success} files trimmed and saved.")
print(f"Output: {OUTPUT_DIR}")