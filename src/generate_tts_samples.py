import os
import numpy as np
import librosa
from scipy.io import wavfile
import requests
import random
import time

SAMPLE_RATE = 16000
COMMANDS = {
    "jelo": "جلو",
    "aghab": "عقب",
    "rast": "راست",
    "chap": "چپ",
    "ist": "ایست"
}
OUTPUT_DIR = "dataset_cleaned_final"
NUM_VARIATIONS = 100
MIN_LENGTH = 1.0

def google_tts(text, output_mp3, lang='fa'):
    """Download TTS from Google Translate"""
    url = "https://translate.google.com/translate_tts"
    params = {
        'ie': 'UTF-8',
        'q': text,
        'tl': lang,
        'client': 'tw-ob',   # seems to work without limits
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        with open(output_mp3, 'wb') as f:
            f.write(response.content)
        return True
    return False

def process_and_save(tmp_mp3, output_wav, speed, pitch_shift):
    audio, sr = librosa.load(tmp_mp3, sr=SAMPLE_RATE, mono=True)
    os.remove(tmp_mp3)
    if speed != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=speed)
    if pitch_shift != 0:
        audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=pitch_shift)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    min_samples = int(MIN_LENGTH * SAMPLE_RATE)
    if len(audio) < min_samples:
        pad_total = min_samples - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), 'constant')
    wavfile.write(output_wav, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return True

def main():
    for cmd in COMMANDS:
        os.makedirs(os.path.join(OUTPUT_DIR, cmd), exist_ok=True)

    for cmd, persian_text in COMMANDS.items():
        print(f"Generating for '{cmd}' ...")
        success = 0
        for i in range(NUM_VARIATIONS):
            speed = np.random.uniform(0.85, 1.15)
            pitch = np.random.uniform(-3, 3)
            tmp_mp3 = f"tmp_google_{cmd}_{i}.mp3"

            # Google Translate TTS is a single voice, but we add speed/pitch variation
            try:
                if google_tts(persian_text, tmp_mp3):
                    wav_path = os.path.join(OUTPUT_DIR, cmd, f"google_{cmd}_{i:03d}.wav")
                    process_and_save(tmp_mp3, wav_path, speed, pitch)
                    success += 1
                    print(f"  {i+1}/{NUM_VARIATIONS} → {os.path.basename(wav_path)}")
                else:
                    print(f"  ❌ Failed sample {i} (HTTP error)")
                # short pause to be polite
                time.sleep(0.3)
            except Exception as e:
                print(f"  Failed sample {i}: {e}")
                if os.path.exists(tmp_mp3):
                    os.remove(tmp_mp3)
        print(f"Done with '{cmd}' (saved {success}/{NUM_VARIATIONS}).\n")

    print("✅ Google TTS generation complete! Now retrain.")

if __name__ == "__main__":
    main()