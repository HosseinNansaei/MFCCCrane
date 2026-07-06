import os
import subprocess
import numpy as np
import librosa
import random
from scipy.io import wavfile

# مسیر کامل espeak-ng
ESPEAK_PATH = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

SAMPLE_RATE = 16000
COMMANDS = {"jelo":"جلو","aghab":"عقب","rast":"راست","chap":"چپ","ist":"ایست"}
OUTPUT_DIR = "dataset_cleaned_final"
NUM = 100

for cmd,text in COMMANDS.items():
    os.makedirs(os.path.join(OUTPUT_DIR,cmd), exist_ok=True)
    for i in range(NUM):
        speed = random.randint(130,180)
        pitch = random.randint(40,60)
        tmp = f"tmp_es_{cmd}.wav"
        
        # استفاده از مسیر کامل
        subprocess.run([ESPEAK_PATH, "-v", "fa", f"-s{speed}", f"-p{pitch}", "-w", tmp, text], check=True)
        
        audio, sr = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
        os.remove(tmp)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio)<0.2*SAMPLE_RATE: 
            continue
        wavfile.write(os.path.join(OUTPUT_DIR,cmd,f"es_{cmd}_{i:03d}.wav"),
                      SAMPLE_RATE, (audio*32767).astype(np.int16))
        print(f"{cmd} {i}")

print("✅ Done")