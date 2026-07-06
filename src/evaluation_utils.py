"""
Voice Command Evaluation Tools
"""
import os
import csv
import time
import json
import random
from datetime import datetime

class CommandTester:
    """Command testing and evaluation class"""
    
    def __init__(self, test_name="test", output_dir="test_results"):
        self.test_name = test_name
        self.output_dir = output_dir
        self.results = []
        self.total_tests = 0
        self.correct_tests = 0
        self.current_test = 0
        
        # Target commands
        self.target_commands = ["jelo", "aghab", "rast", "chap", "ist"]
        self.command_persian = {
            "jelo": "jelo",
            "aghab": "aghab", 
            "rast": "rast",
            "chap": "chap",
            "ist": "ist"
        }
        
        # Create output folder
        os.makedirs(output_dir, exist_ok=True)
        
    def start_test_sequence(self, num_tests=5, manual_files=None):
        """
        Start a test sequence
        manual_files: list of WAV file paths for manual testing
        """
        self.total_tests = num_tests
        self.correct_tests = 0
        self.current_test = 0
        self.results = []
        
        # If manual files provided
        if manual_files and len(manual_files) > 0:
            self.test_mode = "manual_files"
            self.test_sequence = []
            for f in manual_files:
                fname = os.path.basename(f).lower()
                found = False
                for cmd in self.target_commands:
                    if cmd in fname:
                        self.test_sequence.append({'file': f, 'expected': cmd})
                        found = True
                        break
                if not found:
                    self.test_sequence.append({'file': f, 'expected': None})
        else:
            # Auto test - random selection
            self.test_mode = "auto"
            self.test_sequence = random.sample(self.target_commands * 2, num_tests)
            while len(self.test_sequence) < num_tests:
                self.test_sequence.extend(self.target_commands)
            self.test_sequence = self.test_sequence[:num_tests]
            self.test_sequence = [{'expected': cmd, 'file': None} for cmd in self.test_sequence]
        
        print("\n" + "="*60)
        print(f"🧪 Starting test: {self.test_name}")
        print("="*60)
        print(f"📝 Total tests: {len(self.test_sequence)}")
        if self.test_mode == "manual_files":
            print("📂 Mode: Manual WAV file testing")
            for i, item in enumerate(self.test_sequence):
                expected = item['expected'] if item['expected'] else "unknown"
                print(f"   {i+1}. {os.path.basename(item['file'])} -> {expected}")
        else:
            print(f"📋 Expected commands: {[c['expected'] for c in self.test_sequence]}")
        print("="*60 + "\n")
        
        # Create CSV file
        self._init_csv()
        
        return self.test_sequence
    
    def _init_csv(self):
        """Create CSV file with header"""
        self.csv_file = os.path.join(
            self.output_dir, 
            f"{self.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Test_Number', 
                'Expected', 
                'Predicted',
                'Confidence', 
                'Is_Correct',
                'Timestamp',
                'File_Path'
            ])
    
    def record_result(self, expected, predicted, confidence=0, file_path=None):
        """Record a test result"""
        self.current_test += 1
        
        is_correct = (expected == predicted)
        if is_correct:
            self.correct_tests += 1
        
        # Add to results list
        result = {
            'test_number': self.current_test,
            'expected': expected,
            'predicted': predicted if predicted else "unknown",
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_path': file_path or ''
        }
        self.results.append(result)
        
        # Display result
        status = "✅ Correct" if is_correct else "❌ Wrong"
        expected_display = expected
        predicted_display = predicted if predicted else "---"
        
        if self.test_mode == "manual_files" and file_path:
            fname = os.path.basename(file_path)
            print(f"  📁 {fname}: Expected '{expected_display}' -> Predicted '{predicted_display}' {status}")
        else:
            print(f"  Test {self.current_test}/{self.total_tests}: Expected '{expected_display}' -> Predicted '{predicted_display}' {status}")
        
        # Save to CSV
        self._save_to_csv(result)
        
        return is_correct
    
    def _save_to_csv(self, result):
        """Save a result to CSV"""
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result['test_number'],
                result['expected'],
                result['predicted'],
                result['confidence'],
                result['is_correct'],
                result['timestamp'],
                result['file_path']
            ])
    
    def get_accuracy(self):
        """Calculate accuracy"""
        if self.total_tests == 0:
            return 0
        return (self.correct_tests / self.total_tests) * 100
    
    def get_summary(self):
        """Get summary of results"""
        accuracy = self.get_accuracy()
        summary = {
            'test_name': self.test_name,
            'total_tests': self.total_tests,
            'correct_tests': self.correct_tests,
            'accuracy': accuracy,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return summary
    
    def print_summary(self):
        """Print summary of results"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print(f"📊 Test Summary: {self.test_name}")
        print("="*60)
        print(f"✅ Correct tests: {summary['correct_tests']}/{summary['total_tests']}")
        print(f"🎯 Accuracy: {summary['accuracy']:.2f}%")
        print(f"📁 Results file: {self.csv_file}")
        print("="*60 + "\n")
        return summary

def get_user_feedback(expected, predicted, confidence=0):
    """
    Get user feedback
    Returns: (is_correct, correct_command)
    """
    expected_display = expected
    predicted_display = predicted if predicted else "unknown"
    
    print(f"\n{'─'*50}")
    print(f"🎤 Predicted command: {predicted_display} (Confidence: {confidence:.2f})")
    print(f"📌 Expected command: {expected_display}")
    print(f"❓ Is this prediction correct?")
    
    while True:
        answer = input("  (y=yes / n=no / c=change to other command): ").strip().lower()
        
        if answer == 'y':
            return True, None
        elif answer == 'n':
            print("  Select the correct command:")
            print("    1. jelo (forward)")
            print("    2. aghab (backward)")
            print("    3. rast (right)")
            print("    4. chap (left)")
            print("    5. ist (stop)")
            
            while True:
                try:
                    choice = input("  Enter correct command number (1-5): ").strip()
                    cmd_map = {
                        '1': 'jelo',
                        '2': 'aghab',
                        '3': 'rast',
                        '4': 'chap',
                        '5': 'ist'
                    }
                    if choice in cmd_map:
                        return False, cmd_map[choice]
                    else:
                        print("  ⚠️ Please enter a number from 1 to 5.")
                except:
                    pass
        elif answer == 'c':
            print("  Enter the correct command in English:")
            print("  Options: jelo, aghab, rast, chap, ist")
            cmd = input("  Command: ").strip().lower()
            if cmd in ['jelo', 'aghab', 'rast', 'chap', 'ist']:
                return False, cmd
            else:
                print(f"  ⚠️ Command '{cmd}' is not valid.")
        else:
            print("  ⚠️ Please enter y, n, or c.")