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

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512

AUTO_CALIBRATE = True
FIXED_ENERGY_THRESHOLD = 0.005
SILENCE_DURATION = 0.8
SEGMENT_DURATION = 1.5

CONFIDENCE_THRESHOLD = 0.3
SMOOTHING_WINDOW = 5

USE_BANDPASS = True
LOW_CUT = 80
HIGH_CUT = 4000

WIDTH, HEIGHT = 1000, 650
FPS = 60

# Load model
try:
    model = joblib.load("models/sklearn_model_improved.joblib")
    scaler = joblib.load("models/scaler_improved.joblib")
    le = joblib.load("models/label_encoder_improved.joblib")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

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

def bandpass_filter(signal, lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

command_queue = queue.Queue()

def audio_thread():
    global ENERGY_THRESHOLD

    if AUTO_CALIBRATE:
        print("Measuring background noise for 2 seconds...")
        noise_audio = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        noise_energy = np.sqrt(np.mean(noise_audio**2))
        ENERGY_THRESHOLD = max(noise_energy * 3.0, 0.001)  # Minimum threshold
        print(f"Background energy: {noise_energy:.6f} -> threshold: {ENERGY_THRESHOLD:.6f}")
    else:
        ENERGY_THRESHOLD = FIXED_ENERGY_THRESHOLD

    is_speaking = False
    speech_frames = []
    silence_counter = 0
    silence_threshold_frames = int(SILENCE_DURATION * SAMPLE_RATE / 1024)

    def callback(indata, frames, time, status):
        nonlocal is_speaking, speech_frames, silence_counter
        if status:
            if "overflow" in str(status):
                pass  # Ignore overflow warnings
            else:
                print(f"Audio status: {status}")
        
        # Calculate energy
        energy = np.sqrt(np.mean(indata**2))

        if energy > ENERGY_THRESHOLD:
            if not is_speaking:
                print("Speech started...")
                is_speaking = True
                speech_frames = []
            speech_frames.extend(indata[:, 0])
            silence_counter = 0
        else:
            if is_speaking:
                speech_frames.extend(indata[:, 0])
                silence_counter += 1
                if silence_counter > silence_threshold_frames:
                    print("Speech ended, processing...")
                    is_speaking = False
                    silence_counter = 0
                    if len(speech_frames) > 0.3 * SAMPLE_RATE:
                        audio_segment = np.array(speech_frames)

                        if USE_BANDPASS:
                            audio_segment = bandpass_filter(audio_segment, LOW_CUT, HIGH_CUT, SAMPLE_RATE)

                        target_len = int(SEGMENT_DURATION * SAMPLE_RATE)
                        if len(audio_segment) > target_len:
                            frame_energy = librosa.feature.rms(y=audio_segment, frame_length=target_len, hop_length=target_len//4)[0]
                            if len(frame_energy) > 0:
                                best_start = np.argmax(frame_energy) * (target_len//4)
                                best_start = max(0, min(best_start, len(audio_segment)-target_len))
                                audio_segment = audio_segment[best_start:best_start+target_len]
                        elif len(audio_segment) < target_len:
                            pad_total = target_len - len(audio_segment)
                            pad_left = pad_total // 2
                            pad_right = pad_total - pad_left
                            audio_segment = np.pad(audio_segment, (pad_left, pad_right), 'constant')

                        features = extract_features(audio_segment)
                        if features is not None:
                            try:
                                features_scaled = scaler.transform([features])
                                proba = model.predict_proba(features_scaled)[0]
                                pred_id = np.argmax(proba)
                                conf = proba[pred_id]
                                if conf >= CONFIDENCE_THRESHOLD:
                                    command = le.inverse_transform([pred_id])[0]
                                    command_queue.put(command)
                                    print(f"OK: {command} ({conf:.2f})")
                                else:
                                    print(f"Low confidence ({conf:.2f})")
                            except Exception as e:
                                print(f"Error in prediction: {e}")
                    speech_frames = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=1024):
        print("Listening... Speak commands (jelo, aghab, rast, chap, ist)")
        threading.Event().wait()

class Crane:
    def __init__(self):
        self.boom_length = 250
        self.target_boom_length = 250
        self.hook_height = 0.3
        self.target_hook_height = 0.3

        self.boom_min = 120
        self.boom_max = 380
        self.hook_min = 0.05
        self.hook_max = 0.95

        self.base_x = WIDTH // 2
        self.base_y = HEIGHT - 80
        self.tower_height = 120

        self.last_command_time = 0
        self.cooldown = 0.3  # seconds
        
        # Command mapping for display
        self.last_command = "No command"
        self.command_color = (0, 255, 0)

    def apply_command(self, cmd):
        now = time.time()
        if now - self.last_command_time < self.cooldown:
            return
        self.last_command_time = now
        
        # Store last command for display
        cmd_names = {"jelo": "⬆️ جلو", "aghab": "⬇️ عقب", "rast": "➡️ راست", "chap": "⬅️ چپ", "ist": "⏹️ ایست"}
        self.last_command = cmd_names.get(cmd, cmd)

        boom_step = 12
        hook_step = 0.07

        if cmd == 'rast':
            self.target_boom_length = min(self.boom_max, self.target_boom_length + boom_step)
            self.command_color = (0, 255, 0)
        elif cmd == 'chap':
            self.target_boom_length = max(self.boom_min, self.target_boom_length - boom_step)
            self.command_color = (255, 255, 0)
        elif cmd == 'jelo':
            self.target_hook_height = min(self.hook_max, self.target_hook_height + hook_step)
            self.command_color = (0, 255, 255)
        elif cmd == 'aghab':
            self.target_hook_height = max(self.hook_min, self.target_hook_height - hook_step)
            self.command_color = (255, 165, 0)
        elif cmd == 'ist':
            self.target_boom_length = self.boom_length
            self.target_hook_height = self.hook_height
            self.command_color = (255, 0, 0)

    def update(self):
        lerp = 0.12
        self.boom_length += (self.target_boom_length - self.boom_length) * lerp
        self.hook_height += (self.target_hook_height - self.hook_height) * lerp

    def draw(self, screen, font):
        # Ground
        pygame.draw.rect(screen, (0x8a, 0x9a, 0xaa), (0, HEIGHT-60, WIDTH, 60))
        pygame.draw.rect(screen, (0x6a, 0x7e, 0x8e), (0, HEIGHT-55, WIDTH, 5))

        tower_top = self.base_y - self.tower_height
        
        # Tower
        pygame.draw.rect(screen, (0x5a, 0x6a, 0x7a),
                         (self.base_x-20, tower_top, 40, self.tower_height))
        pygame.draw.rect(screen, (0x4a, 0x5a, 0x6a),
                         (self.base_x-15, tower_top, 30, self.tower_height))
        pygame.draw.rect(screen, (0x7a, 0x8a, 0x9a),
                         (self.base_x-35, self.base_y-15, 70, 30))

        # Turntable
        pygame.draw.ellipse(screen, (0x8a, 0x9a, 0xaa),
                            (self.base_x-30, tower_top-12, 60, 24))

        # Boom
        boom_tip_x = self.base_x + self.boom_length
        boom_tip_y = tower_top
        pygame.draw.rect(screen, (0xc9, 0x7e, 0x2a),
                         (self.base_x, boom_tip_y-8, self.boom_length, 16))
        pygame.draw.rect(screen, (0xe8, 0x9e, 0x3a),
                         (self.base_x, boom_tip_y-6, self.boom_length, 12))
        
        # Boom lines
        for i in range(5):
            x = self.base_x + (self.boom_length / 4) * i
            pygame.draw.line(screen, (0xff, 0xcc, 0x66),
                             (x, boom_tip_y-6), (x, boom_tip_y+6), 1)

        # Cable
        cable_length = self.hook_height * 180
        hook_x = boom_tip_x
        hook_y = boom_tip_y + cable_length
        pygame.draw.line(screen, (0x44, 0x44, 0x44),
                         (boom_tip_x, boom_tip_y), (hook_x, hook_y), 2)

        # Hook
        pygame.draw.rect(screen, (0xaa, 0x88, 0x66),
                         (hook_x-8, hook_y-5, 16, 10))
        pygame.draw.circle(screen, (0xcc, 0xaa, 0x77),
                           (hook_x, hook_y+5), 8)
        pygame.draw.circle(screen, (0x88, 0x66, 0x44),
                           (hook_x, hook_y+5), 4)

        # Cabin
        pygame.draw.rect(screen, (0x4a, 0x6a, 0x8a),
                         (self.base_x-25, self.base_y-55, 50, 40))
        pygame.draw.rect(screen, (0xaa, 0xdd, 0xff),
                         (self.base_x-20, self.base_y-50, 40, 30))

        # Info text
        boom_percent = int(((self.boom_length - self.boom_min) / (self.boom_max - self.boom_min)) * 100)
        hook_percent = int(self.hook_height * 100)
        
        # Background for text
        pygame.draw.rect(screen, (255, 255, 255, 200), (10, 10, 350, 100))
        pygame.draw.rect(screen, (200, 200, 200), (10, 10, 350, 100), 2)
        
        text1 = font.render(f"🏗️ Boom: {boom_percent}%  Hook: {hook_percent}%", True, (0, 0, 0))
        screen.blit(text1, (20, 20))
        
        text2 = font.render(f"Last Command: {self.last_command}", True, self.command_color)
        screen.blit(text2, (20, 45))
        
        text3 = font.render(f"Boom: {int(self.boom_length)}px  Hook: {self.hook_height:.2f}", True, (80, 80, 80))
        screen.blit(text3, (20, 70))

def main():
    try:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Voice Controlled Crane - Real Time")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 16)

        crane = Crane()
        print("✅ Pygame window created successfully!")

        # Start audio thread
        t = threading.Thread(target=audio_thread, daemon=True)
        t.start()

        running = True
        while running:
            try:
                cmd = command_queue.get_nowait()
                crane.apply_command(cmd)
            except queue.Empty:
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # Keyboard shortcuts for testing
                    elif event.key == pygame.K_RIGHT:
                        crane.apply_command('rast')
                    elif event.key == pygame.K_LEFT:
                        crane.apply_command('chap')
                    elif event.key == pygame.K_UP:
                        crane.apply_command('jelo')
                    elif event.key == pygame.K_DOWN:
                        crane.apply_command('aghab')
                    elif event.key == pygame.K_SPACE:
                        crane.apply_command('ist')

            crane.update()
            screen.fill((0xdc, 0xe6, 0xf0))
            crane.draw(screen, font)
            
            # Show instructions
            help_font = pygame.font.SysFont("monospace", 12)
            help_text = "Keyboard: ↑↓←→ or Space | ESC to quit"
            help_surface = help_font.render(help_text, True, (100, 100, 100))
            screen.blit(help_surface, (WIDTH - 350, HEIGHT - 30))
            
            pygame.display.flip()
            clock.tick(FPS)

    except Exception as e:
        print(f"❌ Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        print("👋 Pygame closed")

if __name__ == "__main__":
    main()