import os
import numpy as np
import librosa
from scipy.fftpack import dct
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ======================= Settings =======================
DATA_DIR = "dataset_cleaned_final"
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512

# ======================= Full 84‑dimensional feature extraction =======================
def pre_emphasis(signal, coeff=PRE_EMPHASIS):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(signal, sr, fs, stride):
    fl = int(round(fs * sr))
    step = int(round(stride * sr))
    sig_len = len(signal)
    nf = int(np.ceil(float(np.abs(sig_len - fl)) / step)) + 1
    pad = int((nf - 1) * step + fl)
    pad_signal = np.pad(signal, (0, pad - sig_len), 'constant')
    idx = np.tile(np.arange(0, fl), (nf, 1)) + \
          np.tile(np.arange(0, nf * step, step), (fl, 1)).T
    frames = pad_signal[idx.astype(np.int32)]
    frames *= np.hamming(fl)
    return frames

def magspec(frames, nfft):
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
    if nf < 3:
        return np.zeros_like(mfcc)
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
    if len(signal) < int(FRAME_SIZE * SAMPLE_RATE):
        return None
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
    # Prosodic features
    rms = librosa.feature.rms(y=signal, frame_length=int(FRAME_SIZE*SAMPLE_RATE),
                              hop_length=int(FRAME_STRIDE*SAMPLE_RATE))[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=int(FRAME_SIZE*SAMPLE_RATE),
                                             hop_length=int(FRAME_STRIDE*SAMPLE_RATE))[0]
    cent = librosa.feature.spectral_centroid(y=signal, sr=SAMPLE_RATE,
                                             n_fft=NFFT, hop_length=int(FRAME_STRIDE*SAMPLE_RATE))[0]
    for arr in [rms, zcr, cent]:
        feats.append(np.array([np.mean(arr)]))
        feats.append(np.array([np.std(arr)]))
    feat_vec = np.concatenate(feats)
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_vec

# ======================= Data augmentation =======================
def augment_signal(y, sr):
    augmented = [y]
    # Pitch shift
    for steps in [-2, 2]:
        try:
            augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps))
        except:
            pass
    # Time stretch
    for rate in [0.9, 1.1]:
        try:
            augmented.append(librosa.effects.time_stretch(y, rate=rate))
        except:
            pass
    # Add noise at two different strengths
    noise1 = np.random.randn(len(y)) * 0.01
    augmented.append(y + noise1)
    noise2 = np.random.randn(len(y)) * 0.025
    augmented.append(y + noise2)
    return augmented

# ======================= Build dataset =======================
X, y_labels = [], []
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Classes:", class_names)

for cls in class_names:
    cls_path = os.path.join(DATA_DIR, cls)
    files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
    print(f"{cls}: {len(files)} files")
    for file in files:
        fp = os.path.join(cls_path, file)
        try:
            sr, audio = wavfile.read(fp)
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
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            for aug_y in augment_signal(audio, SAMPLE_RATE):
                feat = extract_features(aug_y)
                if feat is not None:
                    X.append(feat)
                    y_labels.append(cls)
        except Exception as e:
            print(f"Error in {fp}: {e}")

X = np.array(X)
print(f"Total samples after augmentation: {len(X)}")

# Remove any rows with NaN (just in case)
nan_mask = np.isnan(X).any(axis=1)
if nan_mask.any():
    print(f"Removing {nan_mask.sum()} NaN samples")
    X = X[~nan_mask]
    y_labels = np.array(y_labels)[~nan_mask]

le = LabelEncoder()
y = le.fit_transform(y_labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# GridSearch for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

model = grid.best_estimator_
acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.2%}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sklearn_model_improved.joblib")
joblib.dump(scaler, "models/scaler_improved.joblib")
joblib.dump(le, "models/label_encoder_improved.joblib")
print("Model saved.")