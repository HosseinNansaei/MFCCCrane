import pygame
import threading
import queue
import warnings
import sounddevice as sd
import joblib
import numpy as np
from scipy.fftpack import dct
from scipy.signal import butter, lfilter
import librosa
import sys
import io
import time
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv
from datetime import datetime

# Fix console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# =========================== SETTINGS ===========================
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512

SEGMENT_DURATION = 1.5
CONFIDENCE_THRESHOLD = 0.3
USE_BANDPASS = True
LOW_CUT = 80
HIGH_CUT = 4000

WIDTH, HEIGHT = 1000, 700

AUTO_RETRAIN_THRESHOLD = 10
CORRECTIONS_FILE = "corrections_data.csv"

MODEL_PATH = "models/sklearn_model_improved.joblib"
SCALER_PATH = "models/scaler_improved.joblib"
ENCODER_PATH = "models/label_encoder_improved.joblib"

# =========================== EVALUATION SYSTEM ===========================
class InteractiveEvaluation:
    def __init__(self):
        self.commands = ["jelo", "aghab", "rast", "chap", "ist"]
        self.results = []
        self.total_tests = 0
        self.correct_tests = 0
        self.command_count = {c: 0 for c in self.commands}
        self.command_correct = {c: 0 for c in self.commands}
        self.test_sequence = []
        self.current_idx = 0
        self.is_evaluating = False
        self.csv_file = None
        self.summary_file = None
        
        os.makedirs("evaluation_results", exist_ok=True)
        
    def generate_sequence(self):
        import random
        self.test_sequence = []
        for cmd in self.commands:
            for _ in range(10):
                self.test_sequence.append(cmd)
        random.shuffle(self.test_sequence)
        self.current_idx = 0
        self.total_tests = len(self.test_sequence)
        self.correct_tests = 0
        self.results = []
        self.command_count = {c: 0 for c in self.commands}
        self.command_correct = {c: 0 for c in self.commands}
        self.is_evaluating = True
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = f"evaluation_results/interactive_{timestamp}.csv"
        self.summary_file = f"evaluation_results/interactive_summary_{timestamp}.csv"
        
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Test_No', 'Expected', 'Predicted', 'Confidence', 'Is_Correct', 'Timestamp'])
        
        print("\n" + "="*60)
        print("🧪 EVALUATION MODE - 50 TESTS (10 per command)")
        print("="*60)
        print(f"📝 Commands: {', '.join([c.upper() for c in self.commands])}")
        print("🎤 Click START, say command, then STOP")
        print("="*60 + "\n")
        
        return self.test_sequence
    
    def get_next(self):
        if self.current_idx < len(self.test_sequence):
            cmd = self.test_sequence[self.current_idx]
            self.current_idx += 1
            return cmd
        self.is_evaluating = False
        return None
    
    def ask_feedback(self, expected, predicted, confidence):
        expected_display = expected.upper()
        predicted_display = predicted.upper() if predicted else "UNKNOWN"
        
        print(f"\n{'─'*50}")
        print(f"🎯 Test #{self.current_idx}/{self.total_tests}")
        print(f"📌 Expected: {expected_display}")
        print(f"🎤 Predicted: {predicted_display} (Confidence: {confidence:.2f})")
        
        while True:
            ans = input("❓ Is this CORRECT? (y=yes / n=no): ").strip().lower()
            if ans == 'y':
                return True
            elif ans == 'n':
                return False
            print("  ⚠️ Please enter 'y' or 'n'")
    
    def record_result(self, expected, predicted, confidence, is_correct):
        self.total_tests += 1
        self.command_count[expected] += 1
        if is_correct:
            self.correct_tests += 1
            self.command_correct[expected] += 1
        
        result = {
            'test_no': self.total_tests,
            'expected': expected,
            'predicted': predicted if predicted else "unknown",
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.results.append(result)
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([result['test_no'], result['expected'], result['predicted'], 
                            result['confidence'], result['is_correct'], result['timestamp']])
    
    def print_summary(self):
        with open(self.summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Command', 'Total', 'Correct', 'Accuracy (%)'])
            for cmd in self.commands:
                total = self.command_count[cmd]
                correct = self.command_correct[cmd]
                acc = (correct / total * 100) if total > 0 else 0
                writer.writerow([cmd, total, correct, f"{acc:.2f}"])
            total_acc = (self.correct_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            writer.writerow(['TOTAL', self.total_tests, self.correct_tests, f"{total_acc:.2f}"])
        
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY - Interactive Method")
        print("="*60)
        print(f"📝 Total tests: {self.total_tests}")
        print(f"✅ Correct: {self.correct_tests}")
        print(f"🎯 Overall Accuracy: {(self.correct_tests/self.total_tests*100):.2f}%" if self.total_tests > 0 else "0%")
        print("-"*60)
        print("📋 Per-Command Accuracy:")
        for cmd in self.commands:
            total = self.command_count[cmd]
            correct = self.command_correct[cmd]
            acc = (correct / total * 100) if total > 0 else 0
            print(f"   {cmd.upper()}: {correct}/{total} ({acc:.2f}%)")
        print("="*60)
        print(f"📁 Results saved to:")
        print(f"   - {self.csv_file}")
        print(f"   - {self.summary_file}")

# =========================== LOAD MODEL ===========================
def load_models():
    global model, scaler, le, class_names
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)
        class_names = le.classes_.tolist()
        print(f"✅ Models loaded! Classes: {class_names}")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

if not load_models():
    exit(1)

# =========================== FEATURE EXTRACTION ===========================
def pre_emphasis(signal, coeff=PRE_EMPHASIS):
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
    try:
        if len(signal) < int(0.025 * SAMPLE_RATE):
            return None
        signal = pre_emphasis(signal)
        frames = framing(signal, SAMPLE_RATE, FRAME_SIZE, FRAME_STRIDE)
        if len(frames) == 0:
            return None
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
    except Exception as e:
        print(f"Extract error: {e}")
        return None

def bandpass_filter(signal, lowcut, highcut, sr, order=5):
    try:
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, signal)
    except:
        return signal

# =========================== BUTTON ===========================
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=(255,255,255), font_size=18):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font_size = font_size
        self.is_hovered = False
        self.enabled = True

    def draw(self, surface, font):
        if not self.enabled:
            color = (50, 50, 50)
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, (255,255,255,100), self.rect, 2, border_radius=10)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# =========================== AUDIO RECORDER ===========================
class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        self.thread = None
        self.stop_flag = False
        self.audio_queue = queue.Queue()
        self.energy_threshold = 0.005
        self.calibrate()

    def calibrate(self):
        try:
            print("Calibrating...")
            noise = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            energy = np.sqrt(np.mean(noise**2))
            self.energy_threshold = max(energy * 3.0, 0.001)
            print(f"Threshold: {self.energy_threshold:.6f}")
        except Exception as e:
            print(f"Calibration error: {e}")
            self.energy_threshold = 0.005

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.stop_flag = False
        self.audio_buffer = []
        self.audio_queue = queue.Queue()
        
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._callback,
            blocksize=1024
        )
        self.stream.start()
        
        self.thread = threading.Thread(target=self._collect_audio, daemon=True)
        self.thread.start()
        print("🎤 Recording started...")

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def _collect_audio(self):
        while not self.stop_flag:
            try:
                data = self.audio_queue.get(timeout=0.1)
                self.audio_buffer.extend(data[:, 0])
            except queue.Empty:
                continue

    def stop_recording(self):
        if not self.is_recording:
            return
        self.stop_flag = True
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("⏹️ Recording stopped.")
        if self.audio_buffer:
            self.audio_data = np.array(self.audio_buffer, dtype=np.float32)
            print(f"✅ Captured: {len(self.audio_data)} samples")
        else:
            self.audio_data = None
            print("❌ No audio captured")

    def get_audio(self):
        return self.audio_data

    def clear_audio(self):
        self.audio_data = None

# =========================== LEARNER ===========================
class InteractiveLearner:
    def __init__(self):
        self.corrections = []
        self.pending_prediction = None
        self.waiting_for_feedback = False
        self.correction_count = 0
        self.load_corrections()

    def load_corrections(self):
        if os.path.exists(CORRECTIONS_FILE):
            try:
                df = pd.read_csv(CORRECTIONS_FILE)
                self.correction_count = len(df)
                print(f"📚 Loaded {self.correction_count} corrections")
            except:
                pass

    def save_correction(self, features, predicted, correct):
        data = {
            'timestamp': time.time(),
            'predicted': predicted,
            'correct': correct,
            'features': ','.join([str(f) for f in features])
        }
        self.corrections.append(data)
        self.correction_count += 1
        pd.DataFrame(self.corrections).to_csv(CORRECTIONS_FILE, index=False)
        print(f"💾 Correction: {predicted} -> {correct}")
        if self.correction_count >= AUTO_RETRAIN_THRESHOLD:
            self.retrain_model()

    def retrain_model(self):
        print("\n🔄 RETRAINING...")
        try:
            csv_file = "features_table_enhanced.csv"
            X_original, y_original = [], []
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                feature_cols = [c for c in df.columns if c not in ['class', 'filename']]
                X_original = df[feature_cols].values.astype(np.float64)
                y_original = df['class'].values

            X_corr, y_corr = [], []
            for corr in self.corrections:
                features = [float(f) for f in corr['features'].split(',')]
                X_corr.append(features)
                y_corr.append(corr['correct'])

            if len(X_corr) == 0 and len(X_original) == 0:
                print("❌ No data to train!")
                return

            if len(X_original) > 0 and len(X_corr) > 0:
                X = np.vstack([X_original, np.array(X_corr)])
                y = np.concatenate([y_original, np.array(y_corr)])
            elif len(X_original) > 0:
                X = X_original
                y = y_original
            else:
                X = np.array(X_corr)
                y = np.array(y_corr)

            if len(X) == 0:
                print("❌ No training data after combining!")
                return

            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            le_new = LabelEncoder()
            y_enc = le_new.fit_transform(y)
            scaler_new = StandardScaler()
            X_scaled = scaler_new.fit_transform(X)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y_enc)

            os.makedirs("models", exist_ok=True)
            joblib.dump(rf, MODEL_PATH)
            joblib.dump(scaler_new, SCALER_PATH)
            joblib.dump(le_new, ENCODER_PATH)

            global model, scaler, le, class_names
            model, scaler, le = rf, scaler_new, le_new
            class_names = le.classes_.tolist()
            print(f"✅ Retrained! Classes: {class_names}")
            self.correction_count = 0
            self.corrections = []
        except Exception as e:
            print(f"❌ Retrain error: {e}")
            import traceback
            traceback.print_exc()

    def start_feedback(self, features, predicted, confidence):
        self.pending_prediction = (features, predicted, confidence)
        self.waiting_for_feedback = True
        return True

    def provide_feedback(self, is_correct, correct_command=None):
        if not self.waiting_for_feedback or self.pending_prediction is None:
            return False
        features, predicted, _ = self.pending_prediction
        if not is_correct and correct_command:
            self.save_correction(features, predicted, correct_command)
        self.waiting_for_feedback = False
        self.pending_prediction = None
        return True

    def is_waiting(self):
        return self.waiting_for_feedback

    def get_predicted_command(self):
        return self.pending_prediction[1] if self.pending_prediction else None

# =========================== CRANE ===========================
class Crane:
    def __init__(self):
        self.boom_length = 250
        self.target_boom_length = 250
        self.hook_height = 0.3
        self.target_hook_height = 0.3
        self.boom_min, self.boom_max = 120, 380
        self.hook_min, self.hook_max = 0.05, 0.95
        self.base_x = WIDTH // 2
        self.base_y = HEIGHT - 150
        self.tower_height = 130
        self.last_command_time = 0
        self.cooldown = 0.3
        self.last_command = "---"
        self.command_color = (100, 200, 255)
        self.last_confidence = 0
        self.status = "MONTZARE FRAMAN..."
        self.status_color = (150, 200, 150)
        self.display_msg = ""
        self.msg_color = (255,255,255)
        self.msg_timer = 0

    def apply_command(self, cmd, confidence=0):
        now = time.time()
        if now - self.last_command_time < self.cooldown:
            return
        self.last_command_time = now
        self.last_confidence = confidence
        cmd_names = {"jelo":"JELO","aghab":"AGHAB","rast":"RAST","chap":"CHAP","ist":"IST"}
        self.last_command = cmd_names.get(cmd, cmd)
        step_b = 15
        step_h = 0.08
        if cmd == 'rast':
            self.target_boom_length = min(self.boom_max, self.target_boom_length + step_b)
            self.command_color = (0,255,100)
        elif cmd == 'chap':
            self.target_boom_length = max(self.boom_min, self.target_boom_length - step_b)
            self.command_color = (255,255,0)
        elif cmd == 'jelo':
            self.target_hook_height = min(self.hook_max, self.target_hook_height + step_h)
            self.command_color = (0,200,255)
        elif cmd == 'aghab':
            self.target_hook_height = max(self.hook_min, self.target_hook_height - step_h)
            self.command_color = (255,165,0)
        elif cmd == 'ist':
            self.target_boom_length = self.boom_length
            self.target_hook_height = self.hook_height
            self.command_color = (255,100,100)

    def set_status(self, msg, color=(200,200,200)):
        self.status = msg
        self.status_color = color

    def set_msg(self, msg, color=(255,255,255), duration=2.0):
        self.display_msg = msg
        self.msg_color = color
        self.msg_timer = time.time() + duration

    def update(self):
        l = 0.12
        self.boom_length += (self.target_boom_length - self.boom_length) * l
        self.hook_height += (self.target_hook_height - self.hook_height) * l

    def draw(self, screen):
        pygame.draw.rect(screen, (50,60,70), (0, HEIGHT-65, WIDTH, 65))
        pygame.draw.rect(screen, (70,80,90), (0, HEIGHT-60, WIDTH, 5))
        tower_top = self.base_y - self.tower_height
        
        pygame.draw.rect(screen, (70,80,90), (self.base_x-22, tower_top, 44, self.tower_height))
        pygame.draw.rect(screen, (90,100,110), (self.base_x-18, tower_top, 36, self.tower_height))
        pygame.draw.rect(screen, (50,60,70), (self.base_x-38, self.base_y-18, 76, 32))
        pygame.draw.ellipse(screen, (100,110,120), (self.base_x-32, tower_top-14, 64, 26))

        boom_tip_x = self.base_x + self.boom_length
        boom_tip_y = tower_top
        pygame.draw.rect(screen, (200,160,60), (self.base_x, boom_tip_y-10, self.boom_length, 20))
        pygame.draw.rect(screen, (230,190,80), (self.base_x, boom_tip_y-7, self.boom_length, 14))
        for i in range(6):
            x = self.base_x + (self.boom_length / 5) * i
            pygame.draw.line(screen, (255,220,120), (x, boom_tip_y-7), (x, boom_tip_y+7), 2)

        cable_length = self.hook_height * 200
        hook_x, hook_y = boom_tip_x, boom_tip_y + cable_length
        pygame.draw.line(screen, (80,80,80), (boom_tip_x, boom_tip_y), (hook_x, hook_y), 3)
        pygame.draw.rect(screen, (180,150,120), (hook_x-10, hook_y-6, 20, 12))
        pygame.draw.circle(screen, (200,180,150), (hook_x, hook_y+6), 10)
        pygame.draw.circle(screen, (150,130,100), (hook_x, hook_y+6), 5)

        pygame.draw.rect(screen, (70,100,130), (self.base_x-28, self.base_y-60, 56, 44))
        pygame.draw.rect(screen, (180,220,255), (self.base_x-22, self.base_y-55, 44, 34))

        bg = pygame.Surface((650, 160))
        bg.set_alpha(220)
        bg.fill((30,35,45))
        screen.blit(bg, (20,20))
        pygame.draw.rect(screen, (80,90,110), (20,20,650,160), 2)

        f_title = pygame.font.Font(None, 28)
        f_small = pygame.font.Font(None, 20)
        f_status = pygame.font.Font(None, 24)

        screen.blit(f_title.render("CRANE - INTERACTIVE MODE", True, (100,200,255)), (35,28))
        boom_pct = int(((self.boom_length - self.boom_min) / (self.boom_max - self.boom_min)) * 100)
        hook_pct = int(self.hook_height * 100)
        screen.blit(f_small.render(f"Boom: {boom_pct}%  Hook: {hook_pct}%  Conf: {self.last_confidence:.2f}", True, (200,200,200)), (35,58))
        screen.blit(f_small.render(f"Last: {self.last_command}", True, self.command_color), (35,82))
        screen.blit(f_status.render(self.status, True, self.status_color), (35,110))

        if evaluator.is_evaluating:
            screen.blit(f_small.render(f"🧪 Evaluation: {evaluator.current_idx}/{evaluator.total_tests}", True, (255,200,0)), (450,58))
        elif learner.is_waiting():
            pred = learner.get_predicted_command()
            if pred:
                screen.blit(f_small.render(f"Predicted: {pred}", True, (255,200,0)), (380,110))

        screen.blit(f_small.render(f"Corrections: {learner.correction_count}/{AUTO_RETRAIN_THRESHOLD}", True, (150,150,150)), (480,82))

        if self.display_msg and time.time() < self.msg_timer:
            screen.blit(pygame.font.Font(None, 22).render(self.display_msg, True, self.msg_color), (35,140))

# =========================== MAIN ===========================
evaluator = InteractiveEvaluation()

def main():
    global learner, recorder, screen, WIDTH, HEIGHT
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Crane - Interactive Learning Mode")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    crane = Crane()
    global learner
    learner = InteractiveLearner()
    recorder = AudioRecorder()

    btn_w, btn_h = 160, 48
    gap = 10
    margin = 20
    start_x = WIDTH - btn_w - margin
    start_y = HEIGHT - 400

    btn_start = Button(start_x, start_y, btn_w, btn_h, "START", (0, 160, 0), (0, 220, 0), font_size=22)
    btn_stop = Button(start_x, start_y + btn_h + gap, btn_w, btn_h, "STOP", (160, 0, 0), (220, 0, 0), font_size=22)
    btn_yes = Button(start_x, start_y + 2*(btn_h + gap), btn_w//2 - 5, btn_h, "YES", (0, 140, 0), (0, 200, 0), font_size=20)
    btn_no = Button(start_x + btn_w//2 + gap, start_y + 2*(btn_h + gap), btn_w//2 - 5, btn_h, "NO", (140, 0, 0), (200, 0, 0), font_size=20)
    
    btn_cor = []
    cor_labels = ["1:jelo", "2:aghab", "3:rast", "4:chap", "5:ist"]
    cor_colors = [(0,100,180), (180,130,0), (0,160,70), (180,90,0), (160,40,40)]
    cor_hcolors = [(0,140,220), (220,170,0), (0,200,90), (220,120,0), (200,60,60)]
    
    cor_w = btn_w//3 - 4
    for i in range(3):
        x = start_x + i * (cor_w + 4)
        y = start_y + 3*(btn_h + gap)
        btn = Button(x, y, cor_w, btn_h, cor_labels[i], cor_colors[i], cor_hcolors[i], font_size=15)
        btn_cor.append(btn)
    
    cor_w2 = btn_w//2 - 4
    for i in range(2):
        x = start_x + i * (cor_w2 + 4)
        y = start_y + 4*(btn_h + gap)
        btn = Button(x, y, cor_w2, btn_h, cor_labels[i+3], cor_colors[i+3], cor_hcolors[i+3], font_size=15)
        btn_cor.append(btn)

    btn_retrain = Button(start_x, start_y + 5*(btn_h + gap) + 10, btn_w, 45, "RETRAIN", (180,150,0), (220,190,0), font_size=18)
    btn_eval = Button(start_x - 200, start_y, 180, btn_h, "EVALUATE (50)", (150, 0, 150), (200, 0, 200), font_size=18)

    for b in [btn_yes, btn_no] + btn_cor:
        b.enabled = False

    pending = None
    running = True
    recording = False
    eval_mode = False

    print("✅ Ready! START to record, STOP to process.")
    print("📋 Click EVALUATE (50) to run 50 tests (10 per command)")

    while running:
        current_w, current_h = pygame.display.get_surface().get_size()
        if current_w != WIDTH or current_h != HEIGHT:
            WIDTH, HEIGHT = current_w, current_h
            crane.base_x = WIDTH // 2
            crane.base_y = HEIGHT - 150
            start_x = WIDTH - btn_w - margin
            start_y = HEIGHT - 400
            
            btn_start.rect.x = start_x
            btn_start.rect.y = start_y
            btn_stop.rect.x = start_x
            btn_stop.rect.y = start_y + btn_h + gap
            btn_yes.rect.x = start_x
            btn_yes.rect.y = start_y + 2*(btn_h + gap)
            btn_no.rect.x = start_x + btn_w//2 + gap
            btn_no.rect.y = start_y + 2*(btn_h + gap)
            
            for i, b in enumerate(btn_cor):
                if i < 3:
                    b.rect.x = start_x + i * (cor_w + 4)
                    b.rect.y = start_y + 3*(btn_h + gap)
                else:
                    b.rect.x = start_x + (i-3) * (cor_w2 + 4)
                    b.rect.y = start_y + 4*(btn_h + gap)
            btn_retrain.rect.x = start_x
            btn_retrain.rect.y = start_y + 5*(btn_h + gap) + 10
            btn_eval.rect.x = start_x - 200
            btn_eval.rect.y = start_y

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_y and learner.is_waiting():
                    learner.provide_feedback(True)
                    crane.set_status("DOROST BUD!", (0,255,100))
                    crane.set_msg("Dorost! YAD GEREFTAM.", (0,255,100), 2.0)
                    for b in [btn_yes, btn_no] + btn_cor:
                        b.enabled = False
                    pending = None
                elif event.key == pygame.K_n and learner.is_waiting():
                    crane.set_status("FRAMAN SAHEH RA ENTEXAB KONID", (255,0,0))
                    crane.set_msg("1-5 entexab konid", (255,0,0), 3.0)
                    btn_yes.enabled = False
                    btn_no.enabled = False
                    for b in btn_cor:
                        b.enabled = True
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    if learner.is_waiting() and pending:
                        mp = {pygame.K_1:'jelo', pygame.K_2:'aghab', pygame.K_3:'rast', pygame.K_4:'chap', pygame.K_5:'ist'}
                        correct = mp[event.key]
                        learner.provide_feedback(False, correct)
                        crane.set_status(f"YAD GEREFTAM: {pending} -> {correct}", (0,200,255))
                        crane.set_msg(f"{pending} -> {correct}", (0,200,255), 3.0)
                        pending = None
                        for b in [btn_yes, btn_no] + btn_cor:
                            b.enabled = False

            if btn_eval.handle_event(event):
                if not eval_mode:
                    evaluator.generate_sequence()
                    eval_mode = True
                    crane.set_status("🧪 EVALUATION MODE - 50 tests", (255,200,0))
                    crane.set_msg("Click START for each command", (255,200,0), 3.0)
                    for b in [btn_yes, btn_no] + btn_cor:
                        b.enabled = False

            if btn_start.handle_event(event):
                if not recorder.is_recording:
                    recorder.start_recording()
                    recording = True
                    crane.set_status("DARHALE ZABT...", (255,200,0))
                    crane.set_msg("Zabt shoro shod...", (255,200,0), 2.0)
                    for b in [btn_yes, btn_no] + btn_cor:
                        b.enabled = False

            if btn_stop.handle_event(event):
                if recorder.is_recording:
                    recorder.stop_recording()
                    recording = False
                    crane.set_status("DARHALE PARDAAZESH...", (0,200,255))
                    
                    audio_data = recorder.get_audio()
                    if audio_data is not None and len(audio_data) > 0.3 * SAMPLE_RATE:
                        audio_seg = audio_data
                        if USE_BANDPASS:
                            audio_seg = bandpass_filter(audio_seg, LOW_CUT, HIGH_CUT, SAMPLE_RATE)
                        
                        target_len = int(SEGMENT_DURATION * SAMPLE_RATE)
                        if len(audio_seg) > target_len:
                            fe = librosa.feature.rms(y=audio_seg, frame_length=target_len, hop_length=target_len//4)[0]
                            if len(fe) > 0:
                                bs = np.argmax(fe) * (target_len//4)
                                bs = max(0, min(bs, len(audio_seg)-target_len))
                                audio_seg = audio_seg[bs:bs+target_len]
                        elif len(audio_seg) < target_len:
                            pad = target_len - len(audio_seg)
                            audio_seg = np.pad(audio_seg, (pad//2, pad - pad//2), 'constant')

                        features = extract_features(audio_seg)
                        if features is not None:
                            try:
                                fs = scaler.transform([features])
                                proba = model.predict_proba(fs)[0]
                                pred_id = np.argmax(proba)
                                conf = proba[pred_id]
                                if conf >= CONFIDENCE_THRESHOLD:
                                    cmd = le.inverse_transform([pred_id])[0]
                                    crane.apply_command(cmd, conf)
                                    
                                    if eval_mode:
                                        expected = evaluator.get_next()
                                        if expected:
                                            is_correct = evaluator.ask_feedback(expected, cmd, conf)
                                            evaluator.record_result(expected, cmd, conf, is_correct)
                                            if is_correct:
                                                crane.set_status(f"✅ Correct! {evaluator.current_idx}/{evaluator.total_tests}", (0,255,100))
                                            else:
                                                crane.set_status(f"❌ Wrong! {evaluator.current_idx}/{evaluator.total_tests}", (255,0,0))
                                            
                                            if evaluator.current_idx >= evaluator.total_tests:
                                                eval_mode = False
                                                evaluator.print_summary()
                                                crane.set_status("✅ Evaluation complete!", (0,255,255))
                                                crane.set_msg("Results saved!", (0,255,255), 3.0)
                                            continue
                                        else:
                                            eval_mode = False
                                            evaluator.print_summary()
                                            crane.set_status("✅ Evaluation complete!", (0,255,255))
                                            continue
                                    
                                    crane.set_status(f"AYA FRAMAN '{cmd}' DOROST AST?", (255,255,0))
                                    learner.start_feedback(features, cmd, conf)
                                    pending = cmd
                                    crane.set_msg(f"Aya '{cmd}' dorost ast? Y/N", (255,255,0), 10.0)
                                    btn_yes.enabled = True
                                    btn_no.enabled = True
                                    for b in btn_cor:
                                        b.enabled = False
                                else:
                                    crane.set_status(f"ETEMAD KAM ({conf:.2f})", (255,100,100))
                                    crane.set_msg("Etemad kam, dobare sabt konid", (255,100,100), 2.0)
                            except Exception as e:
                                crane.set_status("ERROR", (255,0,0))
                                print(f"Prediction error: {e}")
                        else:
                            crane.set_status("EXTRACT ERROR", (255,0,0))
                    else:
                        crane.set_status("SEDA KOTAH", (255,165,0))
                        crane.set_msg("Seda kotah ya daar nadarad", (255,165,0), 2.0)
                    recorder.clear_audio()

            if btn_yes.handle_event(event) and learner.is_waiting():
                learner.provide_feedback(True)
                crane.set_status("DOROST BUD!", (0,255,100))
                crane.set_msg("Dorost! YAD GEREFTAM.", (0,255,100), 2.0)
                for b in [btn_yes, btn_no] + btn_cor:
                    b.enabled = False
                pending = None

            if btn_no.handle_event(event) and learner.is_waiting():
                crane.set_status("FRAMAN SAHEH RA ENTEXAB KONID", (255,0,0))
                crane.set_msg("1-5 entexab konid", (255,0,0), 3.0)
                btn_yes.enabled = False
                btn_no.enabled = False
                for b in btn_cor:
                    b.enabled = True

            for i, b in enumerate(btn_cor):
                if b.handle_event(event) and learner.is_waiting() and pending:
                    correct = ['jelo','aghab','rast','chap','ist'][i]
                    learner.provide_feedback(False, correct)
                    crane.set_status(f"YAD GEREFTAM: {pending} -> {correct}", (0,200,255))
                    crane.set_msg(f"{pending} -> {correct}", (0,200,255), 3.0)
                    pending = None
                    for bb in [btn_yes, btn_no] + btn_cor:
                        bb.enabled = False

            if btn_retrain.handle_event(event):
                learner.retrain_model()
                crane.set_msg("Model Retrained!", (0,255,255), 3.0)
                crane.set_status("RETRAINED", (0,255,255))

        crane.update()
        screen.fill((25,30,40))
        crane.draw(screen)

        btn_start.draw(screen, font)
        btn_stop.draw(screen, font)
        btn_yes.draw(screen, font)
        btn_no.draw(screen, font)
        for b in btn_cor:
            b.draw(screen, font)
        btn_retrain.draw(screen, font)
        btn_eval.draw(screen, font)

        if recording and int(time.time()*2)%2:
            pygame.draw.circle(screen, (255,0,0), (WIDTH-25, HEIGHT-25), 14)
            pygame.draw.circle(screen, (255,0,0), (WIDTH-25, HEIGHT-25), 18, 2)

        pygame.display.flip()
        clock.tick(60)

    if recorder.is_recording:
        recorder.stop_recording()
    pygame.quit()
    print("Goodbye!")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔄 VOICE ROBOT - Interactive Learning Mode")
    print("="*70)
    print("Features:")
    print("  - Click EVALUATE (50) for 50 tests (10 per command)")
    print("  - Normal interactive learning with feedback")
    print("  - Auto-retrain model after 10 corrections")
    print("="*70 + "\n")
    main()