import pygame
import threading
import queue
import json
import time
import sounddevice as sd
import vosk
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SAMPLE_RATE = 16000
MODEL_PATH = "vosk-model-fa-0.5"

# Main commands
COMMANDS = {
    "جلو": "jelo",
    "عقب": "aghab",
    "راست": "rast",
    "چپ": "chap",
    "ایست": "ist"
}

# Aliases for better detection (similar words)
ALIASES = {
    "برو": "jelo",
    "بیا": "jelo",
    "جلوتر": "jelo",
    "عقبتر": "aghab",
    "پشت": "aghab",
    "دور": "aghab",
    "سمت راست": "rast",
    "راستی": "rast",
    "سمت چپ": "chap",
    "چپی": "chap",
    "بس": "ist",
    "توقف": "ist",
    "ایستادن": "ist"
}

COOLDOWN = 0.5

WIDTH, HEIGHT = 1000, 650
FPS = 60

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_closest_command(word, threshold=2):
    """Find the closest command word using Levenshtein distance"""
    # First check exact match in commands
    if word in COMMANDS:
        return COMMANDS[word]
    
    # Check aliases
    if word in ALIASES:
        return ALIASES[word]
    
    # Find closest by Levenshtein distance
    best_match = None
    best_distance = float('inf')
    
    # Check commands
    for cmd_word, cmd_eng in COMMANDS.items():
        dist = levenshtein_distance(word, cmd_word)
        if dist < best_distance and dist <= threshold:
            best_distance = dist
            best_match = cmd_eng
    
    # Check aliases
    if best_match is None:
        for alias, cmd_eng in ALIASES.items():
            dist = levenshtein_distance(word, alias)
            if dist < best_distance and dist <= threshold:
                best_distance = dist
                best_match = cmd_eng
    
    return best_match

model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
recognizer.SetWords(True)

command_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    if recognizer.AcceptWaveform(bytes(indata)):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        if text:
            # Try to display Persian text properly
            try:
                print(f"Recognized: {text}")
            except:
                clean_text = ''.join(c if ord(c) < 128 else '?' for c in text)
                print(f"Recognized: {clean_text}")
            
            # Split text into words
            words = text.split()
            for word in words:
                # Check if word is a command or close to it
                command = find_closest_command(word, threshold=2)
                if command:
                    command_queue.put(command)
                    print(f"Command: {command} (from: {word})")
                    break

def audio_thread():
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000,
                           channels=1, dtype='int16', callback=audio_callback):
        print("Listening... Speak Persian commands (jelo, aghab, rast, chap, ist)")
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

    def apply_command(self, cmd):
        now = time.time()
        if now - self.last_command_time < COOLDOWN:
            return
        self.last_command_time = now

        boom_step = 12
        hook_step = 0.07

        if cmd == 'rast':
            self.target_boom_length = min(self.boom_max, self.target_boom_length + boom_step)
        elif cmd == 'chap':
            self.target_boom_length = max(self.boom_min, self.target_boom_length - boom_step)
        elif cmd == 'jelo':
            self.target_hook_height = min(self.hook_max, self.target_hook_height + hook_step)
        elif cmd == 'aghab':
            self.target_hook_height = max(self.hook_min, self.target_hook_height - hook_step)
        elif cmd == 'ist':
            self.target_boom_length = self.boom_length
            self.target_hook_height = self.hook_height

    def update(self):
        lerp = 0.12
        self.boom_length += (self.target_boom_length - self.boom_length) * lerp
        self.hook_height += (self.target_hook_height - self.hook_height) * lerp

    def draw(self, screen, font):
        pygame.draw.rect(screen, (0x8a, 0x9a, 0xaa), (0, HEIGHT-60, WIDTH, 60))
        pygame.draw.rect(screen, (0x6a, 0x7e, 0x8e), (0, HEIGHT-55, WIDTH, 5))

        tower_top = self.base_y - self.tower_height
        pygame.draw.rect(screen, (0x5a, 0x6a, 0x7a),
                         (self.base_x-20, tower_top, 40, self.tower_height))
        pygame.draw.rect(screen, (0x4a, 0x5a, 0x6a),
                         (self.base_x-15, tower_top, 30, self.tower_height))
        pygame.draw.rect(screen, (0x7a, 0x8a, 0x9a),
                         (self.base_x-35, self.base_y-15, 70, 30))

        pygame.draw.ellipse(screen, (0x8a, 0x9a, 0xaa),
                            (self.base_x-30, tower_top-12, 60, 24))

        boom_tip_x = self.base_x + self.boom_length
        boom_tip_y = tower_top
        pygame.draw.rect(screen, (0xc9, 0x7e, 0x2a),
                         (self.base_x, boom_tip_y-8, self.boom_length, 16))
        pygame.draw.rect(screen, (0xe8, 0x9e, 0x3a),
                         (self.base_x, boom_tip_y-6, self.boom_length, 12))
        for i in range(5):
            x = self.base_x + (self.boom_length / 4) * i
            pygame.draw.line(screen, (0xff, 0xcc, 0x66),
                             (x, boom_tip_y-6), (x, boom_tip_y+6), 1)

        cable_length = self.hook_height * 180
        hook_x = boom_tip_x
        hook_y = boom_tip_y + cable_length
        pygame.draw.line(screen, (0x44, 0x44, 0x44),
                         (boom_tip_x, boom_tip_y), (hook_x, hook_y), 2)

        pygame.draw.rect(screen, (0xaa, 0x88, 0x66),
                         (hook_x-8, hook_y-5, 16, 10))
        pygame.draw.circle(screen, (0xcc, 0xaa, 0x77),
                           (hook_x, hook_y+5), 8)
        pygame.draw.circle(screen, (0x88, 0x66, 0x44),
                           (hook_x, hook_y+5), 4)

        pygame.draw.rect(screen, (0x4a, 0x6a, 0x8a),
                         (self.base_x-25, self.base_y-55, 50, 40))
        pygame.draw.rect(screen, (0xaa, 0xdd, 0xff),
                         (self.base_x-20, self.base_y-50, 40, 30))

        text1 = font.render(f"Boom: {int(self.boom_length)}  Hook: {self.hook_height:.2f}", True, (0,0,0))
        screen.blit(text1, (20, 20))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Voice Controlled Crane")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    crane = Crane()

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

        crane.update()
        screen.fill((0xdc, 0xe6, 0xf0))
        crane.draw(screen, font)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()