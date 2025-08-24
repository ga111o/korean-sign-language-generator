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

# GPU acceleration settings
GPU_CONFIG = {
    'use_gpu': True,  # GPU 사용 여부
    'gpu_device_id': 0,  # 사용할 GPU 디바이스 ID
    'force_cpu': False,  # GPU 실패 시 CPU 강제 사용
    'gpu_memory_limit': 0.8,  # GPU 메모리 사용률 제한 (80%)
    'cuda_enabled': True,  # CUDA 사용 여부
    'tensorrt_enabled': False,  # TensorRT 최적화 (실험적)
}

# MediaPipe settings
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 1,  # 복잡도를 1로 낮춰 안정성 향상
    'enable_segmentation': False,  # 세그멘테이션 비활성화로 오류 방지
    'refine_face_landmarks': True,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# GPU-accelerated MediaPipe settings
MEDIAPIPE_GPU_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 2,  # GPU에서 더 높은 복잡도 사용 가능
    'enable_segmentation': False,
    'refine_face_landmarks': True,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Processing settings
PROCESSING_CONFIG = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'max_num_hands': 2,
    'max_num_faces': 1,
    'min_face_detection_confidence': 0.5
}

# Frame processing settings for stability
FRAME_PROCESSING_CONFIG = {
    'frame_size_tolerance': 0.05,  # 5% 크기 차이 허용
    'max_consecutive_errors': 5,    # 연속 오류 허용 개수
    'resize_interpolation': 'linear',  # 리사이징 보간 방법
    'skip_corrupted_frames': True,     # 손상된 프레임 건너뛰기
    'use_gpu_processing': True,        # GPU 기반 프레임 처리 사용
    'gpu_batch_size': 8,               # GPU 배치 처리 크기
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
