# 🏗️ MFCCCrane

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Librosa-0.8%2B-red?style=for-the-badge&logo=python&logoColor=white" alt="Librosa">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensource&logoColor=white" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <b>Persian Voice Command Recognition System for Industrial Cranes</b><br>
  <i>سیستم تشخیص فرمان‌های صوتی فارسی برای کنترل جرثقیل صنعتی</i>
</p>

---

## 📖 Table of Contents

- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Benchmark Results](#-benchmark-results)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Evaluation Results](#-evaluation-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About The Project

**MFCCCrane** is an advanced Persian voice command recognition system designed specifically for **industrial crane control**. The system leverages **MFCC (Mel-Frequency Cepstral Coefficients)** features combined with machine learning algorithms to accurately recognize five essential Persian commands:

| Command | Persian | Meaning |
|---------|---------|---------|
| `jelo` | جلو | Forward |
| `aghab` | عقب | Backward |
| `rast` | راست | Right |
| `chap` | چپ | Left |
| `ist` | ایست | Stop |

The system supports **4 different implementation methods**, includes a **comprehensive evaluation framework**, and generates **professional benchmark reports** for academic and industrial use.

---

## ✨ Key Features

### 🎤 Voice Recognition
- Real-time and file-based voice command detection
- MFCC + Delta + Delta-Delta feature extraction (84-dimensional vector)
- Confidence-based prediction with threshold filtering
- Bandpass denoising filter (80–4000 Hz) for enhanced accuracy

### 🤖 Implementation Methods
| Method | Input | Description |
|--------|-------|-------------|
| **File-Based** | WAV files | Process pre-recorded audio files |
| **Real-Time** | Microphone | Live voice command detection |
| **Vosk** | Microphone | Persian ASR with Vosk model |
| **Interactive** | Microphone | Active learning with user feedback |

### 📊 Evaluation & Benchmarking
- **50 tests per method** with automated validation
- **7 evaluation categories**:
  1. Feature Extraction Methods
  2. Denoising Techniques
  3. Outlier Removal
  4. Classification Models
  5. Data Augmentation
  6. Implementation Methods
  7. Feature Dimensions
- **LaTeX report generation** with professional tables and charts

### 🖥️ Web Interface
- Real-time crane simulation
- Persian command display
- Visual feedback with confidence scores
- Interactive crane boom and hook control

### 🔄 Interactive Learning
- User feedback integration
- Automatic model retraining after 10 corrections
- Persistent correction storage (`corrections_data.csv`)

---

## 🏗️ Architecture

```
        ┌─────────────────────────────────────────────┐
        │          VOICE COMMAND RECOGNITION          │
        │              MFCCCrane System               │
        └─────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼─────────────────────┐
        │                      │                     │
   ┌────▼────┐          ┌─────▼─────┐        ┌───────▼───────┐
   │ Audio   │          │ Feature   │        │ Classification│
   │ Input   │──────────▶Extraction ───────▶│   Models      │
   │         │          │  (MFCC)   │        │               │
   └─────────┘          └───────────┘        └───────────────┘
        │                                           │
        │                                           │
   ┌────▼────┐                              ┌───────▼───────┐
   │ WAV     │                              │  Random Forest│
   │ Micro   │                              │  SVM          │
   │ Vosk    │                              │  KNN          │
   │ Stream  │                              │  GridSearch   │
   └─────────┘                              └───────────────┘
```

### Feature Pipeline

```
Raw Audio → Pre-emphasis → Framing → FFT → Mel Filter Bank → DCT → MFCC
                                                                    ↓
                                                          Delta + Delta-Delta
                                                                    ↓
                                                          Prosodic Features
                                                                    ↓
                                                         84-dim Feature Vector
```

---

## 📊 Benchmark Results

### Category-wise Best Results

| Category | Best Method | Score | Metric |
|----------|-------------|-------|--------|
| 🎵 Feature Extraction | MFCC+Delta (13 dim) | **100.0%** | Accuracy |
| 🧹 Denoising | No Denoising | **0.37 dB** | SNR Improvement |
| 📊 Outlier Removal | None | **100.0%** | Accuracy |
| 🤖 Classification | Random Forest | **100.0%** | Accuracy |
| 📈 Data Augmentation | All Combined | **98.0%** | Accuracy |
| ⚙️ Implementation | File-Based (WAV) | **96.4%** | Accuracy |
| 📏 Feature Dimensions | 13 dims | **100.0%** | Accuracy |
| 🧪 Evaluation (50 Tests) | Vosk | **62.0%** | Accuracy |

### Classification Models Comparison

| Model | Accuracy | Train Time | Predict Time |
|-------|----------|------------|--------------|
| 🥇 Random Forest | **100.0%** | 0.581s | 0.064s |
| 🥇 SVM | **100.0%** | 0.063s | 0.016s |
| 🥇 RF (GridSearch) | **100.0%** | 3.340s | 0.062s |
| Simple KNN | 98.7% | 0.000s | 0.004s |
| Weighted KNN | 98.7% | 0.000s | 0.013s |
| KNN (sklearn) | 98.7% | 0.000s | 0.345s |

### Feature Dimension Analysis

| Dimensions | Accuracy | Description |
|------------|----------|-------------|
| 13 | **100.0%** | MFCC only |
| 26 | **100.0%** | MFCC + Delta |
| 39 | **100.0%** | MFCC + Delta + Delta-Delta |
| 78 | **100.0%** | Complete features (without prosodic) |
| 84 | **100.0%** | Complete features (with prosodic) |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/HosseinNansaei/MFCCCrane.git
cd MFCCCrane

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the main menu
python src/project_manager.py
```

### Quick Commands

```bash
# Train the model
python src/train_model_sklearn_improved.py

# Extract features from dataset
python src/extract_features_enhanced.py

# Run evaluation (all methods)
python src/evaluate_all.py

# Run full benchmark
python src/Global_research.py

# Launch web interface
python src/voice_robot.py
```

---

## 📁 Project Structure

```
MFCCCrane/
│
├── 📄 README.md                 # Project documentation
├── 📄 LICENSE                   # MIT License
├── 📄 requirements.txt          # Python dependencies
├── 📄 .gitignore                # Git ignore rules
│
├── 📁 src/                      # Source code
│   ├── __init__.py
│   ├── project_manager.py       # Main menu interface
│   ├── train_model_sklearn_improved.py
│   ├── extract_features_enhanced.py
│   ├── evaluate_all.py          # Complete evaluation system
│   ├── voice_robot.py           # File-based robot
│   ├── voice_robot_realtime_improved.py
│   ├── voice_robot_vosk.py
│   ├── voice_robot_Interactive_Learning_Mode.py
│   ├── voice_robot_evaluation.py
│   ├── Global_research.py       # Full benchmark
│   ├── benchmark_models.py
│   ├── auto_remove_outliers_enhanced.py
│   ├── filter_dataset_by_prediction.py
│   ├── clean_dataset_enhanced.py
│   ├── record_new_samples.py
│   ├── generate_tts_samples.py
│   ├── ModelDL.py
│   ├── evaluation_utils.py
│   └── utils.py
│
├── 📁 models/                   # Trained models
│   ├── sklearn_model_improved.joblib
│   ├── scaler_improved.joblib
│   ├── label_encoder_improved.joblib
│   └── voice_model_improved.npz
│
├── 📁 data/                     # Dataset and features
│   ├── dataset_cleaned_final.zip   # Cleaned audio files
│   │   ├── jelo/
│   │   ├── aghab/
│   │   ├── rast/
│   │   ├── chap/
│   │   └── ist/
│   ├── data.zip   
│   ├── features_table_enhanced.csv
│   └── corrections_data.csv
│
├── 📁 web/                      # Web interface
│   └── index.html
│
└── 📁 results/                  # Benchmark and evaluation results
    ├── benchmark_tables.rar
    │   
    │   
    │   
    └── evaluation_results.rar


    
```

---

## 🛠️ Technologies

| Technology       | Purpose                                           |
|------------------|---------------------------------------------------|
| **Python 3.8+**  | Core programming language                         |
| **scikit-learn** | Machine learning models (Random Forest, SVM, KNN) |
| **Librosa**      | Audio processing and MFCC extraction              |
| **NumPy/SciPy**  | Numerical computations and signal processing      |
| **Pygame**       | GUI and crane visualization                       |
| **SoundDevice**  | Real-time audio recording                         |
| **Matplotlib**   | Benchmark charts and graphs                       |
| **Vosk**         | Persian speech recognition                        |
| **Joblib**       |  Model persistence                                |

---

## 📈 Evaluation Results (50 Tests per Method)

| Method | Accuracy |
|--------|----------|
| Vosk | 62.0% |
| Interactive Learning | 62.0% |
| File-Based (WAV) | 40.0% |
| Real-Time | 24.0% |

### Per-Command Accuracy (File-Based)

| Command | Correct / Total | Accuracy |
|---------|-----------------|----------|
| `aghab` | 10/10 | 100.0% |
| `chap` | 10/10 | 100.0% |
| `jelo` | 0/10 | 0.0% |
| `rast` | 0/10 | 0.0% |
| `ist` | 0/10 | 0.0% |

> **Note:** The file-based evaluation used default dataset files; some commands had no matching files, resulting in 0% accuracy for those commands. The evaluation system is designed to test any custom WAV files.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🌟 Acknowledgments

- [Librosa](https://librosa.org/) - Audio processing library
- [scikit-learn](https://scikit-learn.org/) - Machine learning tools
- [Vosk](https://alphacephei.com/vosk/) - Persian speech recognition
- [Shields.io](https://shields.io/) - Badge generation

---

<p align="center">
  <b>Made with ❤️ for Industrial Crane Control with Voice</b>
</p>
