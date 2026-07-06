import os
import csv
import numpy as np
from scipy.fftpack import dct
from scipy.signal import resample
from scipy.io import wavfile
import warnings
import librosa

warnings.filterwarnings("ignore")

DATA_DIR = "dataset_cleaned_final"
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(signal, sr, fs, stride):
    fl = int(round(fs * sr))
    step = int(round(stride * sr))
    sig_len = len(signal)
    nf = int(np.ceil(float(np.abs(sig_len - fl)) / step)) + 1
    pad = int((nf - 1) * step + fl)
    pad_signal = np.pad(signal, (0, pad - sig_len), 'constant')
    idx = np.tile(np.arange(0, fl), (nf, 1)) + np.tile(np.arange(0, nf * step, step), (fl, 1)).T
    frames = pad_signal[idx.astype(np.int32)]
    frames *= np.hamming(fl)
    return frames

def magspec(frames, nfft=512):
    mag = np.absolute(np.fft.rfft(frames, nfft))
    return (mag ** 2) / nfft

def mel_filterbank(nfilt, nfft, sr):
    low_mel = 2595 * np.log10(1 + 0 / 700)
    high_mel = 2595 * np.log10(1 + (sr/2) / 700)
    mel_pts = np.linspace(low_mel, high_mel, nfilt+2)
    hz_pts = 700 * (10**(mel_pts/2595) - 1)
    bin = np.floor((nfft+1) * hz_pts / sr).astype(int)
    fbank = np.zeros((nfilt, int(np.floor(nfft/2+1))))
    for m in range(1, nfilt+1):
        fm1, fm, fm2 = bin[m-1], bin[m], bin[m+1]
        for k in range(fm1, fm):
            fbank[m-1, k] = (k - fm1) / (fm - fm1)
        for k in range(fm, fm2):
            fbank[m-1, k] = (fm2 - k) / (fm2 - fm)
    return fbank

def compute_delta(mfcc, N=2):
    nf = mfcc.shape[0]
    delta = np.zeros_like(mfcc)
    for t in range(nf):
        numerator = np.zeros(mfcc.shape[1])
        denom = 0
        for n in range(1, N+1):
            if t+n < nf and t-n >= 0:
                numerator += n * (mfcc[t+n] - mfcc[t-n])
                denom += n**2
        delta[t] = numerator / (2*denom) if denom > 0 else 0
    return delta

def extract_features(signal):
    signal = pre_emphasis(signal)
    frames = framing(signal, SAMPLE_RATE, FRAME_SIZE, FRAME_STRIDE)
    pow_frames = magspec(frames, NFFT)
    fbank = mel_filterbank(26, NFFT, SAMPLE_RATE)
    fb = np.dot(pow_frames, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)
    fb = 20 * np.log10(fb)
    mfcc = dct(fb, type=2, axis=1, norm='ortho')[:, :N_MFCC]
    d1 = compute_delta(mfcc, 2)
    d2 = compute_delta(d1, 2)
    feats = []
    for feat in [mfcc, d1, d2]:
        feats.append(np.mean(feat, axis=0))
        feats.append(np.std(feat, axis=0))
    rms = librosa.feature.rms(y=signal, frame_length=int(FRAME_SIZE*SAMPLE_RATE),
                              hop_length=int(FRAME_STRIDE*SAMPLE_RATE))[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=int(FRAME_SIZE*SAMPLE_RATE),
                                             hop_length=int(FRAME_STRIDE*SAMPLE_RATE))[0]
    cent = librosa.feature.spectral_centroid(y=signal, sr=SAMPLE_RATE,
                                             n_fft=NFFT, hop_length=int(FRAME_STRIDE*SAMPLE_RATE))[0]
    for arr in [rms, zcr, cent]:
        feats.append(np.array([np.mean(arr)]))
        feats.append(np.array([np.std(arr)]))
    return np.concatenate(feats)

csv_file = "features_table_enhanced.csv"
feature_names = []
for prefix in ['mfcc', 'delta1', 'delta2']:
    for stat in ['mean', 'std']:
        for i in range(N_MFCC):
            feature_names.append(f"{prefix}_{stat}_{i+1}")
for name in ['rms', 'zcr', 'centroid']:
    feature_names.append(f"{name}_mean")
    feature_names.append(f"{name}_std")

print("Extracting features and saving to CSV...")

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['class', 'filename'] + feature_names)
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    for cls in class_names:
        cls_path = os.path.join(DATA_DIR, cls)
        for file in os.listdir(cls_path):
            if not file.endswith('.wav'):
                continue
            filepath = os.path.join(cls_path, file)
            try:
                sr, audio = wavfile.read(filepath)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                elif audio.dtype != np.float32:
                    max_val = np.max(np.abs(audio))
                    audio = audio.astype(np.float32) / max_val if max_val > 0 else audio
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != SAMPLE_RATE:
                    audio = resample(audio, int(len(audio)*SAMPLE_RATE/sr))
                features = extract_features(audio)
                writer.writerow([cls, file] + features.tolist())
            except Exception as e:
                print(f"Error in {filepath}: {e}")

print(f"CSV saved: {csv_file}")