import os
import sys
import time
import csv
import random
import threading
import queue
import json
import warnings
from datetime import datetime

import numpy as np
import joblib
import librosa
import sounddevice as sd
import pygame

from scipy.fftpack import dct
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# =========================== SETTINGS ===========================
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512
CONFIDENCE_THRESHOLD = 0.3

# =========================== LOAD MODEL ===========================
def load_models():
    try:
        model = joblib.load("models/sklearn_model_improved.joblib")
        scaler = joblib.load("models/scaler_improved.joblib")
        le = joblib.load("models/label_encoder_improved.joblib")
        return model, scaler, le
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

model, scaler, le = load_models()
if model is None:
    print("Models not found! Run train_model_sklearn_improved.py first.")
    sys.exit(1)

class_names = le.classes_.tolist()
print(f"✅ Models loaded! Classes: {class_names}")

# =========================== FEATURE EXTRACTION ===========================
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
    if len(signal) < int(0.025 * SAMPLE_RATE):
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
        if feat.shape[0] == 0:
            return None
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
    feat_vec = np.concatenate(feats)
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_vec

def bandpass_filter(signal, lowcut=80, highcut=4000, sr=SAMPLE_RATE, order=5):
    """Denoise filter - bandpass 80-4000 Hz"""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def predict_command(features):
    features_scaled = scaler.transform([features])
    proba = model.predict_proba(features_scaled)[0]
    pred_id = np.argmax(proba)
    confidence = float(proba[pred_id])
    command = le.inverse_transform([pred_id])[0]
    return command, confidence

# =========================== EVALUATION BASE CLASS ===========================
class EvaluationBase:
    def __init__(self, method_name):
        self.method_name = method_name
        self.commands = ["jelo", "aghab", "rast", "chap", "ist"]
        self.results = []
        self.total_tests = 0
        self.correct_tests = 0
        self.command_count = {c: 0 for c in self.commands}
        self.command_correct = {c: 0 for c in self.commands}
        self.test_sequence = []
        self.current_idx = 0
        self.is_evaluating = False
        os.makedirs("evaluation_results", exist_ok=True)

    def generate_sequence(self):
        self.test_sequence = []
        for cmd in self.commands:
            for _ in range(10):
                self.test_sequence.append(cmd)
        random.shuffle(self.test_sequence)
        self.current_idx = 0
        self.total_tests = len(self.test_sequence)
        self.correct_tests = 0
        self.is_evaluating = True
        print("\n" + "="*60)
        print(f"EVALUATION - {self.method_name} - 50 TESTS (10 per command)")
        print("="*60)
        print(f"Commands: {', '.join([c.upper() for c in self.commands])}")
        print("="*60 + "\n")
        return self.test_sequence

    def get_next(self):
        if self.current_idx < self.total_tests:
            cmd = self.test_sequence[self.current_idx]
            self.current_idx += 1
            return cmd
        self.is_evaluating = False
        return None

    def record_result(self, expected, predicted, confidence, is_correct):
        test_no = self.current_idx
        self.command_count[expected] += 1
        if is_correct:
            self.correct_tests += 1
            self.command_correct[expected] += 1
        self.results.append({
            'test_no': test_no,
            'expected': expected,
            'predicted': predicted if predicted else "UNKNOWN",
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        self._save_to_csv()

    def _save_to_csv(self):
        csv_file = f"evaluation_results/{self.method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Test_No', 'Expected', 'Predicted', 'Confidence', 'Is_Correct', 'Timestamp'])
            for r in self.results:
                writer.writerow([r['test_no'], r['expected'], r['predicted'], r['confidence'],
                                 r['is_correct'], r['timestamp']])
        summary_file = f"evaluation_results/{self.method_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Command', 'Total', 'Correct', 'Accuracy (%)'])
            for cmd in self.commands:
                total = self.command_count[cmd]
                correct = self.command_correct[cmd]
                acc = (correct / total * 100) if total > 0 else 0
                writer.writerow([cmd, total, correct, f"{acc:.2f}"])
            total_acc = (self.correct_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            writer.writerow(['TOTAL', self.total_tests, self.correct_tests, f"{total_acc:.2f}"])

    def print_summary(self):
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY - {self.method_name}")
        print("="*60)
        print(f"Total tests: {self.total_tests}")
        print(f"Correct: {self.correct_tests}")
        print(f"Overall Accuracy: {(self.correct_tests/self.total_tests*100):.2f}%" if self.total_tests > 0 else "0%")
        print("-"*60)
        print("Per-Command Accuracy:")
        for cmd in self.commands:
            total = self.command_count[cmd]
            correct = self.command_correct[cmd]
            acc = (correct / total * 100) if total > 0 else 0
            print(f"   {cmd.upper()}: {correct}/{total} ({acc:.2f}%)")
        print("="*60)

# =========================== 1. FILE-BASED EVALUATION ===========================
def evaluate_file_based():
    eval_obj = EvaluationBase("filebased")
    eval_obj.generate_sequence()
    
    print("📂 Enter the path to WAV files for each command:")
    print("   (e.g., dataset_cleaned_final/jelo/jelo_001.wav)")
    print("   Press Enter to use default dataset files")
    wav_files = {}
    for cmd in eval_obj.commands:
        default_path = f"dataset_cleaned_final/{cmd}"
        print(f"\n  {cmd.upper()}:")
        choice = input(f"    Use default? (y=yes / n=enter custom path): ").strip().lower()
        if choice == 'y':
            wav_files[cmd] = default_path
        else:
            path = input(f"    Enter path: ").strip()
            wav_files[cmd] = path if path else default_path
    
    print("\n" + "="*60)
    print("Starting evaluation... (AUTO mode - no user input needed)")
    print("="*60 + "\n")
    
    for _ in range(50):
        expected = eval_obj.get_next()
        if expected is None:
            break
        search_dir = wav_files.get(expected, f"dataset_cleaned_final/{expected}")
        wav_file = None
        if os.path.exists(search_dir):
            if os.path.isdir(search_dir):
                files = [f for f in os.listdir(search_dir) if f.endswith('.wav')]
                if files:
                    wav_file = os.path.join(search_dir, files[0])
            else:
                wav_file = search_dir
        if wav_file and os.path.exists(wav_file):
            try:
                sr, audio = wavfile.read(wav_file)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    max_val = np.max(np.abs(audio))
                    audio = audio.astype(np.float32) / max_val if max_val > 0 else audio
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                # Apply denoise filter
                audio = bandpass_filter(audio)
                features = extract_features(audio)
                if features is not None:
                    predicted, confidence = predict_command(features)
                    is_correct = (expected == predicted)
                    eval_obj.record_result(expected, predicted, confidence, is_correct)
                    status = "✅" if is_correct else "❌"
                    print(f"  Test #{eval_obj.current_idx}: Expected={expected.upper()} Predicted={predicted.upper()} {status}")
                else:
                    print(f"  ⚠️ Feature extraction failed for {wav_file}")
                    eval_obj.record_result(expected, None, 0, False)
            except Exception as e:
                print(f"  ❌ Error: {e}")
                eval_obj.record_result(expected, None, 0, False)
        else:
            print(f"  ⚠️ No WAV file found for {expected}")
            eval_obj.record_result(expected, None, 0, False)
    
    eval_obj.print_summary()

# =========================== 2. REAL-TIME EVALUATION ===========================
def evaluate_realtime():
    eval_obj = EvaluationBase("realtime")
    eval_obj.generate_sequence()
    
    print("🎤 Real-Time Evaluation - AUTO mode")
    print("Speak each command when prompted (you have 5 seconds)\n")
    
    # Calibration (like original code)
    print("Calibrating background noise for 2 seconds...")
    noise = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    noise_energy = np.sqrt(np.mean(noise**2))
    energy_threshold = max(noise_energy * 3.0, 0.001)
    print(f"Energy threshold: {energy_threshold:.6f}")
    
    for _ in range(50):
        expected = eval_obj.get_next()
        if expected is None:
            break
        print(f"\n{'─'*50}")
        print(f"Test #{eval_obj.current_idx}/{eval_obj.total_tests}")
        print(f"Say: {expected.upper()}")
        print("Speak now... (5 seconds)")
        
        # Record audio (like original realtime code)
        audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        
        audio = audio.flatten()
        # Check energy
        energy = np.sqrt(np.mean(audio**2))
        if energy < energy_threshold or len(audio) < 0.3 * SAMPLE_RATE:
            print("  ⏱️ No speech detected (too quiet or too short)")
            eval_obj.record_result(expected, None, 0, False)
            continue
        
        # Apply denoise filter
        audio = bandpass_filter(audio)
        
        # Trim to segment duration (like original)
        target_len = int(1.5 * SAMPLE_RATE)
        if len(audio) > target_len:
            frame_energy = librosa.feature.rms(y=audio, frame_length=target_len, hop_length=target_len//4)[0]
            if len(frame_energy) > 0:
                best_start = np.argmax(frame_energy) * (target_len//4)
                best_start = max(0, min(best_start, len(audio)-target_len))
                audio = audio[best_start:best_start+target_len]
        elif len(audio) < target_len:
            pad = target_len - len(audio)
            audio = np.pad(audio, (pad//2, pad - pad//2), 'constant')
        
        features = extract_features(audio)
        if features is not None:
            predicted, confidence = predict_command(features)
            is_correct = (expected == predicted)
            eval_obj.record_result(expected, predicted, confidence, is_correct)
            status = "✅" if is_correct else "❌"
            print(f"  Predicted: {predicted.upper()} {status} (conf: {confidence:.2f})")
        else:
            print("  ⚠️ Feature extraction failed")
            eval_obj.record_result(expected, None, 0, False)
    
    eval_obj.print_summary()

# =========================== 3. VOSK EVALUATION ===========================
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

def evaluate_vosk():
    if not VOSK_AVAILABLE:
        print("❌ Vosk not installed! Install with: pip install vosk")
        return
    
    eval_obj = EvaluationBase("vosk")
    eval_obj.generate_sequence()
    
    vosk_model_path = "vosk-model-fa-0.5"
    if not os.path.exists(vosk_model_path):
        print(f"❌ Vosk model not found at {vosk_model_path}")
        return
    
    model_vosk = vosk.Model(vosk_model_path)
    recognizer = vosk.KaldiRecognizer(model_vosk, SAMPLE_RATE)
    recognizer.SetWords(True)
    
    persian_map = {"جلو": "jelo", "عقب": "aghab", "راست": "rast", "چپ": "chap", "ایست": "ist"}
    
    print("🎙️ Vosk Evaluation - AUTO mode")
    print("Say each command in Persian when prompted\n")
    
    for _ in range(50):
        expected = eval_obj.get_next()
        if expected is None:
            break
        persian = {v: k for k, v in persian_map.items()}.get(expected, expected)
        print(f"\n{'─'*50}")
        print(f"Test #{eval_obj.current_idx}/{eval_obj.total_tests}")
        print(f"Say: {persian} ({expected.upper()})")
        print("Speak now... (5 seconds)")
        
        # Record audio for Vosk (5 seconds)
        audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        audio = audio.flatten()
        
        # Process with Vosk
        if recognizer.AcceptWaveform(audio.tobytes()):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                predicted = None
                for word in text.split():
                    if word in persian_map:
                        predicted = persian_map[word]
                        break
                if predicted:
                    is_correct = (expected == predicted)
                    eval_obj.record_result(expected, predicted, 0.5, is_correct)
                    status = "✅" if is_correct else "❌"
                    print(f"  Predicted: {predicted.upper()} {status} (from: '{text}')")
                else:
                    print(f"  ⚠️ No command found in: '{text}'")
                    eval_obj.record_result(expected, None, 0, False)
            else:
                print("  ⏱️ No speech recognized")
                eval_obj.record_result(expected, None, 0, False)
        else:
            print("  ⏱️ No speech detected")
            eval_obj.record_result(expected, None, 0, False)
    
    eval_obj.print_summary()

# =========================== 4. INTERACTIVE EVALUATION (uses real-time) ===========================
def evaluate_interactive():
    print("🔄 Interactive Evaluation - Using Real-Time method (with GUI would be similar)")
    print("For simplicity, using the same Real-Time evaluation method.\n")
    evaluate_realtime()

# =========================== MAIN MENU ===========================
def show_menu():
    print("\n" + "="*70)
    print("📊 VOICE COMMAND EVALUATION SYSTEM (AUTO mode)")
    print("="*70)
    print("All evaluations are automatic - no y/n questions!")
    print("The system compares Expected vs Predicted automatically.")
    print("✅ Denoise filter (bandpass 80-4000 Hz) is applied to all audio.")
    print("="*70)
    print("Options:")
    print("  1. File-Based (WAV) - 50 tests")
    print("  2. Real-Time (Microphone) - 50 tests")
    print("  3. Vosk (Persian ASR) - 50 tests")
    print("  4. Interactive (Real-Time with GUI) - 50 tests")
    print("  5. Run ALL methods (Full comparison)")
    print("  0. Exit")
    print("="*70)

if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("\nSelect option (0-5): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            evaluate_file_based()
            input("\nPress Enter to continue...")
        elif choice == '2':
            evaluate_realtime()
            input("\nPress Enter to continue...")
        elif choice == '3':
            evaluate_vosk()
            input("\nPress Enter to continue...")
        elif choice == '4':
            evaluate_interactive()
            input("\nPress Enter to continue...")
        elif choice == '5':
            print("\n" + "█"*70)
            print("RUNNING ALL EVALUATIONS...")
            print("█"*70)
            evaluate_file_based()
            evaluate_realtime()
            evaluate_vosk()
            evaluate_interactive()
            print("\n" + "█"*70)
            print("✅ ALL EVALUATIONS COMPLETE!")
            print("📁 Results saved to: evaluation_results/")
            print("█"*70)
            input("\nPress Enter to continue...")
        else:
            print("❌ Invalid option!")