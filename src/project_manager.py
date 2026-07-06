"""
PROJECT MANAGER - Main Menu for SRB_Project
"""

import os
import sys
import subprocess
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color=Colors.GREEN):
    print(f"{color}{text}{Colors.END}")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_name, description):
    print_colored(f"\n▶️ Running: {description}", Colors.BLUE)
    print_colored(f"📄 Script: {script_name}", Colors.YELLOW)
    print("-" * 60)
    print()
    
    # Run with real-time output
    try:
        process = subprocess.Popen(
            [sys.executable, script_name],
            cwd=PROJECT_ROOT,
            stdout=None,  # Use parent's stdout
            stderr=None,  # Use parent's stderr
            stdin=None,   # Use parent's stdin
            text=True
        )
        process.wait()
    except Exception as e:
        print_colored(f"❌ Error: {e}", Colors.RED)
    
    input("\nPress Enter to continue...")

def open_html():
    html_path = os.path.join(PROJECT_ROOT, "index.html")
    if os.path.exists(html_path):
        print_colored(f"\n🌐 Opening: {html_path}", Colors.BLUE)
        webbrowser.open(html_path)
    else:
        print_colored("\n❌ index.html not found!", Colors.RED)
    input("\nPress Enter to continue...")

def main_menu():
    while True:
        clear_screen()
        print("=" * 70)
        print_colored("🚀 SRB_PROJECT - MAIN MENU", Colors.HEADER + Colors.BOLD)
        print("=" * 70)
        print_colored("📂 Project Root: " + PROJECT_ROOT, Colors.YELLOW)
        print("-" * 70)
        
        print_colored("\n📊 TRAINING AND DATA PREPARATION", Colors.BOLD)
        print("  [1] Train Model (sklearn_improved)")
        print("  [2] Extract Features (Enhanced)")
        print("  [3] Remove Outliers (Auto)")
        print("  [4] Filter Dataset by Prediction")
        print("  [5] Clean Dataset")
        print("  [6] Record New Samples")
        print("  [7] Generate TTS Samples")
        
        print_colored("\n🤖 VOICE ROBOT IMPLEMENTATIONS", Colors.BOLD)
        print("  [8] File-Based Robot (WAV)")
        print("  [9] Real-Time Robot (Voice)")
        print("  [10] Vosk Robot (Pre-trained)")
        print("  [11] Interactive Learning Mode")
        print("  [12] Deep Learning Model (DL)")
        print("  [13] Open Web Interface (index.html)")
        
        print_colored("\n🧪 EVALUATION AND TESTING", Colors.BOLD)
        print("  [14] Run All Evaluations (evaluate_all.py)")
        print("  [15] File-Based Evaluation")
        print("  [16] Real-Time Evaluation")
        print("  [17] Vosk Evaluation")
        print("  [18] Interactive Evaluation")
        
        print_colored("\n📊 BENCHMARK AND ANALYSIS", Colors.BOLD)
        print("  [19] Full Benchmark (Global Research)")
        print("  [20] Model Comparison (Benchmark)")
        
        print_colored("\n📁 PROJECT ORGANIZATION", Colors.BOLD)
        print("  [21] Organize Project for Submission")
        
        print_colored("\n🔧 UTILITIES", Colors.BOLD)
        print("  [22] View Project Summary")
        print("  [23] Open Project Folder")
        
        print_colored("\n" + "-" * 70, Colors.YELLOW)
        print("  [0] Exit")
        print("-" * 70)
        
        choice = input("\n📌 Enter your choice: ").strip()
        
        scripts = {
            '1': ("train_model_sklearn_improved.py", "Training Model"),
            '2': ("extract_features_enhanced.py", "Extracting Features"),
            '3': ("auto_remove_outliers_enhanced.py", "Removing Outliers"),
            '4': ("filter_dataset_by_prediction.py", "Filtering Dataset"),
            '5': ("clean_dataset_enhanced.py", "Cleaning Dataset"),
            '6': ("record_new_samples.py", "Recording New Samples"),
            '7': ("generate_tts_samples.py", "Generating TTS Samples"),
            '8': ("voice_robot.py", "File-Based Robot (WAV)"),
            '9': ("voice_robot_realtime_improved.py", "Real-Time Robot"),
            '10': ("voice_robot_vosk.py", "Vosk Robot"),
            '11': ("voice_robot_Interactive_Learning_Mode.py", "Interactive Learning"),
            '12': ("ModelDL.py", "Deep Learning Model"),
            '13': ("open_html", "Web Interface"),
            '14': ("evaluate_all.py", "All Evaluations"),
            '15': ("evaluate_all.py", "File-Based Evaluation"),
            '16': ("evaluate_all.py", "Real-Time Evaluation"),
            '17': ("evaluate_all.py", "Vosk Evaluation"),
            '18': ("evaluate_all.py", "Interactive Evaluation"),
            '19': ("Global_research.py", "Full Benchmark"),
            '20': ("benchmark_models.py", "Model Comparison"),
            '21': ("organize_project.py", "Organizing Project")
        }
        
        if choice == '0':
            print_colored("\n👋 Goodbye!", Colors.GREEN)
            break
        
        elif choice == '13':
            open_html()
        
        elif choice == '22':
            print_colored("\n📋 Project Summary", Colors.BLUE)
            print("-" * 60)
            print("Project: SRB_Project")
            print("Status: Active")
            print(f"Location: {PROJECT_ROOT}")
            print("\n📁 Key Files:")
            print("  - evaluate_all.py: Complete evaluation system")
            print("  - voice_robot_Interactive_Learning_Mode.py: Interactive learning")
            print("  - Global_research.py: Full benchmark")
            print("  - benchmark_models.py: Model comparison")
            print("\n📊 Evaluation Results:")
            eval_dir = "evaluation_results"
            if os.path.exists(eval_dir):
                summary_files = [f for f in os.listdir(eval_dir) if f.endswith('_summary.csv')]
                for sf in summary_files:
                    print(f"  - {sf}")
            input("\nPress Enter to continue...")
        
        elif choice == '23':
            print_colored(f"\n📂 Opening: {PROJECT_ROOT}", Colors.BLUE)
            if os.name == 'nt':
                os.startfile(PROJECT_ROOT)
            else:
                subprocess.run(['open', PROJECT_ROOT])
            input("\nPress Enter to continue...")
        
        elif choice in scripts:
            script_name, description = scripts[choice]
            script_path = os.path.join(PROJECT_ROOT, script_name)
            
            # Special handling for evaluation options
            if choice in ['14', '15', '16', '17', '18']:
                # Pass option to evaluate_all.py
                eval_options = {
                    '14': '5',  # All evaluations
                    '15': '1',  # File-Based
                    '16': '2',  # Real-Time
                    '17': '3',  # Vosk
                    '18': '4'   # Interactive
                }
                print_colored(f"\n▶️ Running: {description}", Colors.BLUE)
                print_colored(f"📄 Script: {script_name} (option {eval_options[choice]})", Colors.YELLOW)
                print("-" * 60)
                print()
                try:
                    process = subprocess.Popen(
                        [sys.executable, script_name, eval_options[choice]],
                        cwd=PROJECT_ROOT,
                        stdout=None,
                        stderr=None,
                        stdin=None,
                        text=True
                    )
                    process.wait()
                except Exception as e:
                    print_colored(f"❌ Error: {e}", Colors.RED)
                input("\nPress Enter to continue...")
            elif os.path.exists(script_path):
                run_script(script_path, description)
            else:
                print_colored(f"\n❌ Script not found: {script_name}", Colors.RED)
                input("\nPress Enter to continue...")
        
        else:
            print_colored("\n❌ Invalid choice! Please try again.", Colors.RED)
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print_colored("\n\n👋 Goodbye!", Colors.GREEN)
    except Exception as e:
        print_colored(f"\n❌ Error: {e}", Colors.RED)
        input("\nPress Enter to exit...")