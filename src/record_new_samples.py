import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import librosa
import os

SAMPLE_RATE = 16000
RECORD_DURATION = 5.0       # max recording length (you can stop early)
TOP_DB = 20                 # silence threshold for trimming

COMMANDS = ["jelo", "aghab", "rast", "chap", "ist"]
DATASET_DIR = "dataset_cleaned_final"

def record_audio():
    """Record audio, return float32 mono array"""
    input("Press ENTER to start recording (max 3 sec, Ctrl+C to stop early)...")
    try:
        audio = sd.rec(int(RECORD_DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        print("🔴 Recording... speak clearly!")
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\n⏹️ Recording stopped early.")
        audio = sd.get_recording()
    return audio.flatten()

def play_audio(audio):
    """Play audio array (float32, 16000 Hz)"""
    print("🔊 Playing...")
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()
    print("🔇 Playback finished.")

def trim_silence(audio):
    """Trim leading/trailing silence, return trimmed array"""
    trimmed, _ = librosa.effects.trim(audio, top_db=TOP_DB)
    return trimmed

def main():
    print("=" * 50)
    print("NEW DATASET RECORDER (with preview)")
    print("=" * 50)

    while True:
        # Choose command
        print("\nAvailable commands:")
        for i, cmd in enumerate(COMMANDS, 1):
            print(f"  {i}. {cmd}")
        choice = input("Choose command number (or type name, 'q' to quit): ").strip().lower()
        if choice == 'q':
            break
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(COMMANDS):
                command = COMMANDS[idx]
            else:
                print("❌ Invalid number.")
                continue
        elif choice in COMMANDS:
            command = choice
        else:
            print("❌ Invalid command.")
            continue

        # Record → trim → play → decide
        while True:
            raw = record_audio()
            trimmed = trim_silence(raw)

            if len(trimmed) < 0.2 * SAMPLE_RATE:
                print("⚠️ Audio too short (less than 0.2 sec). Please record again.")
                continue

            play_audio(trimmed)

            decision = input("✅ Save this? (y = save, n = discard & re-record, p = play again): ").strip().lower()
            if decision == 'y':
                # Save
                folder = os.path.join(DATASET_DIR, command)
                os.makedirs(folder, exist_ok=True)
                existing = [f for f in os.listdir(folder) if f.endswith('.wav')]
                count = len(existing) + 1
                filename = f"{command}_{count:03d}.wav"
                output_path = os.path.join(folder, filename)
                wavfile.write(output_path, SAMPLE_RATE, (trimmed * 32767).astype(np.int16))
                print(f"💾 Saved: {output_path}")
                break
            elif decision == 'n':
                print("🗑️ Discarded. Please record again.")
                # loop back to record again
            elif decision == 'p':
                play_audio(trimmed)
                # then ask again
            else:
                print("🤔 Please answer 'y', 'n', or 'p'.")

        again = input("\nRecord another command? (y/n): ").strip().lower()
        if again != 'y':
            print("👋 Recording finished. Run extract_features and retrain.")
            break

if __name__ == "__main__":
    main()