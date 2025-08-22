"""
Configuration file for Sign Language Skeleton Extractor
Centralized configuration for all paths, settings, and parameters
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs" / "extractor"

# Input files
CSV_FILENAME = "signlanguage_total.csv"
CSV_PATH = DATA_DIR / CSV_FILENAME

KNET_WORKSTATION_DATASET_DIR = Path("/mnt/f/signlanguage/dataset")
LOCAL_DATASET_DIR = Path("./dataset")

DATASET_PATHS = [
    KNET_WORKSTATION_DATASET_DIR, 
    BASE_DIR / "dataset", 
    Path("./dataset"), 
]

# Output settings
DEFAULT_OUTPUT_DIR = OUTPUT_DIR

# Logging settings
LOG_FILENAME_PREFIX = "sign_language_processing"

# Video processing settings
SUPPORTED_VIDEO_EXTENSIONS = [
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mts',
    '.MOV', '.MTS'  
]

# MediaPipe settings
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 2,
    'enable_segmentation': True,
    'refine_face_landmarks': True
}

# Processing settings
PROCESSING_CONFIG = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'max_num_hands': 2,
    'max_num_faces': 1,
    'min_face_detection_confidence': 0.5
}

# File naming patterns
OUTPUT_FILENAME_PATTERN = "{base_name}_skeleton_data.json"

# Filter settings
DEFAULT_DIRECTION_FILTER = "정면"

# Performance settings
BATCH_SIZE = 32
MAX_WORKERS = 4

# Debug settings
DEBUG_MODE = False
SAVE_INTERMEDIATE_FRAMES = False
INTERMEDIATE_FRAMES_DIR = OUTPUT_DIR / "intermediate_frames"
