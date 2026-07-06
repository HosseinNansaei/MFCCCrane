"""
MFCCCrane - Persian Voice Command Recognition System
=====================================================

A machine learning-based system for recognizing Persian voice commands
(jelo, aghab, rast, chap, ist) to control industrial cranes.

This package includes:
- Feature extraction (MFCC, Delta, Delta-Delta, Prosodic features)
- Multiple classification models (RF, SVM, KNN, Weighted KNN)
- 4 implementation methods (File-Based, Real-Time, Vosk, Interactive)
- Comprehensive evaluation and benchmarking framework
- Web-based crane visualization interface

Author: SRB Project Team
License: MIT
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "SRB Project Team"
__license__ = "MIT"
__description__ = "Voice command recognition system for industrial cranes using MFCC features"

# Package metadata for PyPI compatibility (optional)
__all__ = [
    "project_manager",
    "voice_robot",
    "voice_robot_realtime_improved",
    "voice_robot_vosk",
    "voice_robot_Interactive_Learning_Mode",
    "Global_research",
    "benchmark_models",
    "evaluate_all",
    "train_model_sklearn_improved",
    "extract_features_enhanced",
]

# Optional: Import key modules to make them available at package level
# from . import project_manager
# from . import voice_robot

print(f"🔊 MFCCCrane v{__version__} loaded successfully!")
