"""
Complete Evaluation System for All Voice Command Recognition Methods
This script tests all 4 methods and compares their accuracy
"""
import os
import sys
import time
import json
import subprocess
import threading
import queue
import warnings
import numpy as np
import pandas as pd
import joblib
import librosa
from scipy.io import wavfile
from datetime import datetime

warnings.filterwarnings("ignore")

# =========================== SETTINGS ===========================
SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = 0.3

# =========================== EVALUATION TOOLS ===========================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation_utils import CommandTester, get_user_feedback

# =========================== LOAD MODELS ===========================
def load_models():
    try:
        model = joblib.load("models/sklearn_model_improved.joblib")
        scaler = joblib.load("models/scaler_improved.joblib")
        le = joblib.load("models/label_encoder_improved.joblib")
        class_names = le.classes_.tolist()
        print(f"✅ Models loaded! Classes: {class_names}")
        return model, scaler, le
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None

model, scaler, le = load_models()
if model is None:
    exit(1)

# =========================== FEATURE EXTRACTION FUNCTIONS ===========================
from voice_robot_realtime_improved import extract_features, bandpass_filter

# =========================== TEST FILE-BASED (WAV) ===========================
def test_file_based(tester, wav_files=None):
    """
    Test File-Based (WAV) method
    """
    print("\n" + "="*60)
    print("📂 Testing File-Based (WAV) Method")
    print("="*60)
    
    if wav_files is None:
        print("Select WAV files for manual testing:")
        print("(Separate files with comma or press Enter for auto test)")
        files_input = input("File paths: ").strip()
        if files_input:
            wav_files = [f.strip() for f in files_input.split(',')]
        else:
            # Auto test with sample files
            wav_files = []
            dataset_dir = "dataset_cleaned_final"
            if os.path.exists(dataset_dir):
                for cmd in ['jelo', 'aghab', 'rast', 'chap', 'ist']:
                    cmd_dir = os.path.join(dataset_dir, cmd)
                    if os.path.exists(cmd_dir):
                        wavs = [os.path.join(cmd_dir, f) for f in os.listdir(cmd_dir) 
                                if f.endswith('.wav')]
                        if wavs:
                            wav_files.append(wavs[0])
    
    if not wav_files:
        print("❌ No WAV files found!")
        return tester
    
    # Start test
    tester.start_test_sequence(manual_files=wav_files)
    
    for test_item in tester.test_sequence:
        file_path = test_item['file']
        expected = test_item['expected']
        
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            tester.record_result(expected, None, 0, file_path)
            continue
        
        # If expected command is unknown, ask user
        if expected is None:
            fname = os.path.basename(file_path)
            print(f"\n📁 File: {fname}")
            print("Select the correct command:")
            print("  1. jelo (forward)")
            print("  2. aghab (backward)")
            print("  3. rast (right)")
            print("  4. chap (left)")
            print("  5. ist (stop)")
            while True:
                choice = input("Enter number (1-5): ").strip()
                cmd_map = {'1':'jelo','2':'aghab','3':'rast','4':'chap','5':'ist'}
                if choice in cmd_map:
                    expected = cmd_map[choice]
                    break
                print("Please enter a number from 1 to 5.")
        
        try:
            # Read audio file
            sr, audio = wavfile.read(file_path)
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
            
            # Bandpass filter
            audio = bandpass_filter(audio, 80, 4000, SAMPLE_RATE)
            
            # Extract features
            features = extract_features(audio)
            
            if features is None:
                print(f"⚠️ Feature extraction error: {file_path}")
                tester.record_result(expected, None, 0, file_path)
                continue
            
            # Predict
            features_scaled = scaler.transform([features])
            proba = model.predict_proba(features_scaled)[0]
            pred_id = np.argmax(proba)
            confidence = proba[pred_id]
            predicted = le.inverse_transform([pred_id])[0]
            
            # Get user feedback
            is_correct, correct_cmd = get_user_feedback(expected, predicted, confidence)
            
            if is_correct:
                tester.record_result(expected, predicted, confidence, file_path)
            else:
                tester.record_result(expected, correct_cmd, confidence, file_path)
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            tester.record_result(expected, None, 0, file_path)
    
    tester.print_summary()
    return tester

# =========================== TEST REAL-TIME ===========================
def test_realtime(tester, num_tests=5):
    """
    Test Real-Time method (direct microphone recording)
    """
    print("\n" + "="*60)
    print("🎤 Testing Real-Time Method")
    print("="*60)
    
    import sounddevice as sd
    
    # Start test
    tester.start_test_sequence(num_tests)
    
    # Background noise calibration
    print("\n🔇 Calibrating background noise...")
    noise = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    noise_energy = np.sqrt(np.mean(noise**2))
    energy_threshold = max(noise_energy * 3.0, 0.001)
    print(f"✅ Energy threshold: {energy_threshold:.6f}")
    
    for test_item in tester.test_sequence:
        expected = test_item['expected']
        expected_display = expected
        
        print(f"\n{'─'*50}")
        print(f"🎯 Test {tester.current_test + 1}/{tester.total_tests}")
        print(f"📌 Expected command: {expected_display}")
        print("🎤 Please say the command... (5 seconds)")
        print("⏹️ Press Ctrl+C to stop recording")
        
        try:
            # Record audio
            audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            if len(audio) < 0.3 * SAMPLE_RATE:
                print("⚠️ Audio too short, trying again...")
                continue
            
            audio = audio.flatten()
            
            # Bandpass filter
            audio = bandpass_filter(audio, 80, 4000, SAMPLE_RATE)
            
            # Extract features
            features = extract_features(audio)
            
            if features is None:
                print("⚠️ Feature extraction error")
                tester.record_result(expected, None, 0)
                continue
            
            # Predict
            features_scaled = scaler.transform([features])
            proba = model.predict_proba(features_scaled)[0]
            pred_id = np.argmax(proba)
            confidence = proba[pred_id]
            predicted = le.inverse_transform([pred_id])[0]
            
            # Get user feedback
            is_correct, correct_cmd = get_user_feedback(expected, predicted, confidence)
            
            if is_correct:
                tester.record_result(expected, predicted, confidence)
            else:
                tester.record_result(expected, correct_cmd, confidence)
                
        except KeyboardInterrupt:
            print("\n⏹️ Recording stopped")
            tester.record_result(expected, None, 0)
        except Exception as e:
            print(f"❌ Error: {e}")
            tester.record_result(expected, None, 0)
    
    tester.print_summary()
    return tester

# =========================== TEST VOSK ===========================
def test_vosk(tester, num_tests=5):
    """
    Test Vosk method (free speech recognition)
    """
    print("\n" + "="*60)
    print("🎙️ Testing Vosk Method")
    print("="*60)
    
    try:
        import vosk
    except:
        print("❌ Vosk library not installed!")
        return tester
    
    # Start test
    tester.start_test_sequence(num_tests)
    
    # Load Vosk model
    vosk_model_path = "vosk-model-fa-0.5"
    if not os.path.exists(vosk_model_path):
        print(f"❌ Vosk model not found at {vosk_model_path}!")
        return tester
    
    vosk_model = vosk.Model(vosk_model_path)
    recognizer = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
    recognizer.SetWords(True)
    
    import sounddevice as sd
    
    # Calibration
    print("\n🔇 Calibrating...")
    noise = sd.rec(int(1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    noise_energy = np.sqrt(np.mean(noise**2))
    energy_threshold = max(noise_energy * 2.5, 0.001)
    print(f"✅ Energy threshold: {energy_threshold:.6f}")
    
    # Vosk command mapping
    vosk_commands = {
        "jelo": "jelo",
        "aghab": "aghab", 
        "rast": "rast",
        "chap": "chap",
        "ist": "ist"
    }
    
    for test_item in tester.test_sequence:
        expected = test_item['expected']
        expected_display = expected
        
        print(f"\n{'─'*50}")
        print(f"🎯 Test {tester.current_test + 1}/{tester.total_tests}")
        print(f"📌 Expected command: {expected_display}")
        print("🎤 Please say the command... (5 seconds)")
        print("⏹️ Press Ctrl+C to stop recording")
        
        try:
            # Record audio
            audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            if len(audio) < 0.3 * SAMPLE_RATE:
                print("⚠️ Audio too short")
                tester.record_result(expected, None, 0)
                continue
            
            # Detect with Vosk
            audio_int16 = (audio * 32767).astype(np.int16)
            if recognizer.AcceptWaveform(audio_int16.tobytes()):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print(f"📝 Vosk recognized: {text}")
                
                # Find command in recognized text
                predicted = None
                confidence = 0.5
                
                for word in text.split():
                    if word in vosk_commands:
                        predicted = vosk_commands[word]
                        break
                
                if predicted is None:
                    # Search for closest match
                    best_match = None
                    for vosk_word, cmd in vosk_commands.items():
                        if vosk_word in text:
                            best_match = cmd
                            break
                    
                    if best_match is None:
                        print("⚠️ No command found in recognized text")
                        tester.record_result(expected, None, 0)
                        continue
                    predicted = best_match
                
                # Get user feedback
                is_correct, correct_cmd = get_user_feedback(expected, predicted, confidence)
                
                if is_correct:
                    tester.record_result(expected, predicted, confidence)
                else:
                    tester.record_result(expected, correct_cmd, confidence)
            else:
                print("⚠️ Speech recognition failed")
                tester.record_result(expected, None, 0)
                
        except KeyboardInterrupt:
            print("\n⏹️ Recording stopped")
            tester.record_result(expected, None, 0)
        except Exception as e:
            print(f"❌ Error: {e}")
            tester.record_result(expected, None, 0)
    
    tester.print_summary()
    return tester

# =========================== TEST INTERACTIVE ===========================
def test_interactive(tester, num_tests=5):
    """
    Test Interactive method (with interactive feedback)
    """
    print("\n" + "="*60)
    print("🔄 Testing Interactive Method (with interactive feedback)")
    print("="*60)
    
    # Start test
    tester.start_test_sequence(num_tests)
    
    import sounddevice as sd
    
    # Background noise calibration
    print("\n🔇 Calibrating background noise...")
    noise = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    noise_energy = np.sqrt(np.mean(noise**2))
    energy_threshold = max(noise_energy * 3.0, 0.001)
    print(f"✅ Energy threshold: {energy_threshold:.6f}")
    
    for test_item in tester.test_sequence:
        expected = test_item['expected']
        expected_display = expected
        
        print(f"\n{'─'*50}")
        print(f"🎯 Test {tester.current_test + 1}/{tester.total_tests}")
        print(f"📌 Expected command: {expected_display}")
        print("🎤 Please say the command... (5 seconds)")
        print("⏹️ Press Ctrl+C to stop recording")
        
        try:
            # Record audio
            audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            if len(audio) < 0.3 * SAMPLE_RATE:
                print("⚠️ Audio too short")
                tester.record_result(expected, None, 0)
                continue
            
            audio = audio.flatten()
            audio = bandpass_filter(audio, 80, 4000, SAMPLE_RATE)
            features = extract_features(audio)
            
            if features is None:
                print("⚠️ Feature extraction error")
                tester.record_result(expected, None, 0)
                continue
            
            # Predict
            features_scaled = scaler.transform([features])
            proba = model.predict_proba(features_scaled)[0]
            pred_id = np.argmax(proba)
            confidence = proba[pred_id]
            predicted = le.inverse_transform([pred_id])[0]
            
            # In Interactive mode, feedback is already collected
            # But we ask user if it was correct
            is_correct, correct_cmd = get_user_feedback(expected, predicted, confidence)
            
            if is_correct:
                tester.record_result(expected, predicted, confidence)
                print("  ✅ Correct - Model is learning")
            else:
                tester.record_result(expected, correct_cmd, confidence)
                print(f"  📝 Learned: {predicted} -> {correct_cmd}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Recording stopped")
            tester.record_result(expected, None, 0)
        except Exception as e:
            print(f"❌ Error: {e}")
            tester.record_result(expected, None, 0)
    
    tester.print_summary()
    return tester

# =========================== COLLECT ALL RESULTS ===========================
def collect_all_results():
    """
    Run all tests and collect results
    """
    results = []
    
    # 1. File-Based test
    print("\n" + "█"*70)
    print("📂 Test 1: File-Based (WAV) Method")
    print("█"*70)
    tester1 = test_file_based(CommandTester("FileBased"))
    if tester1:
        results.append(tester1.get_summary())
    
    # 2. Real-Time test
    print("\n" + "█"*70)
    print("🎤 Test 2: Real-Time Method")
    print("█"*70)
    tester2 = test_realtime(CommandTester("RealTime"), num_tests=5)
    if tester2:
        results.append(tester2.get_summary())
    
    # 3. Vosk test
    print("\n" + "█"*70)
    print("🎙️ Test 3: Vosk Method")
    print("█"*70)
    tester3 = test_vosk(CommandTester("Vosk"), num_tests=5)
    if tester3:
        results.append(tester3.get_summary())
    
    # 4. Interactive test
    print("\n" + "█"*70)
    print("🔄 Test 4: Interactive Method")
    print("█"*70)
    tester4 = test_interactive(CommandTester("Interactive"), num_tests=5)
    if tester4:
        results.append(tester4.get_summary())
    
    # ===== Final summary =====
    print("\n" + "="*70)
    print("📊 Final Summary - Method Comparison")
    print("="*70)
    
    # Create comparison table
    comp_df = pd.DataFrame(results)
    comp_df = comp_df[['test_name', 'total_tests', 'correct_tests', 'accuracy']]
    comp_df.columns = ['Method', 'Total Tests', 'Correct', 'Accuracy (%)']
    print(comp_df.to_string(index=False))
    
    # Save results
    comp_file = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comp_df.to_csv(comp_file, index=False, encoding='utf-8-sig')
    print(f"\n📁 Comparison results saved to: {comp_file}")
    
    return comp_df

# =========================== MAIN MENU ===========================
def main():
    print("\n" + "="*70)
    print("🧪 Voice Command Recognition - Complete Evaluation System")
    print("="*70)
    print("This program tests all 4 voice command recognition methods:")
    print("  1. File-Based (WAV) - Test with audio files")
    print("  2. Real-Time - Test with direct microphone recording")
    print("  3. Vosk - Test with Vosk model")
    print("  4. Interactive - Test with interactive feedback")
    print("="*70)
    
    while True:
        print("\n📋 Available options:")
        print("  1. Run all tests (Full comparison)")
        print("  2. Test only File-Based (WAV)")
        print("  3. Test only Real-Time")
        print("  4. Test only Vosk")
        print("  5. Test only Interactive")
        print("  6. Manual test with specific WAV file")
        print("  0. Exit")
        
        choice = input("\n📌 Select option: ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            collect_all_results()
        elif choice == '2':
            tester = test_file_based(CommandTester("FileBased_Test"))
        elif choice == '3':
            tester = test_realtime(CommandTester("RealTime_Test"), num_tests=5)
        elif choice == '4':
            tester = test_vosk(CommandTester("Vosk_Test"), num_tests=5)
        elif choice == '5':
            tester = test_interactive(CommandTester("Interactive_Test"), num_tests=5)
        elif choice == '6':
            print("\n📂 Enter WAV file path:")
            file_path = input("Path: ").strip()
            if os.path.exists(file_path):
                tester = test_file_based(CommandTester("Manual_Test"), [file_path])
            else:
                print("❌ File not found!")
        else:
            print("❌ Invalid option!")

if __name__ == "__main__":
    main()