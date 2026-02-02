"""
Configuration file for Driver Drowsiness Detection preprocessing
"""
import os

# ===================================================================
# PATH CONFIGURATION
# ===================================================================
# Dataset paths
BASE_DIR = r"D:\PROJECTDriverDrowsiness"
DATASET_DIR = os.path.join(BASE_DIR, "FaceImages")
ACTIVE_DIR = os.path.join(DATASET_DIR, "ActiveSubjects")
FATIGUE_DIR = os.path.join(DATASET_DIR, "FatigueSubjects")

# Output paths for preprocessed data
OUTPUT_DIR = os.path.join(BASE_DIR, "PreprocessedData")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_TRAIN_FILE = os.path.join(OUTPUT_DIR, "X_train.npy")
X_TEST_FILE = os.path.join(OUTPUT_DIR, "X_test.npy")
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "y_train.npy")
Y_TEST_FILE = os.path.join(OUTPUT_DIR, "y_test.npy")

# Cache files (để tránh phải xử lý lại toàn bộ)
X_CACHE_FILE = os.path.join(OUTPUT_DIR, "X_preprocessed.npy")
Y_CACHE_FILE = os.path.join(OUTPUT_DIR, "y_preprocessed.npy")

# ===================================================================
# MODEL CONFIGURATION
# ===================================================================
IMG_SIZE = 145
CATEGORIES = ["FatigueSubjects", "ActiveSubjects"]  # 0: Fatigue, 1: Active
CATEGORY_MAPPING = {
    "Fatigue Subjects": "FatigueSubjects",
    "Active Subjects": "ActiveSubjects"
}

# ===================================================================
# MEDIAPIPE LANDMARKS CONFIGURATION
# ===================================================================
# Eye landmark indices (sẽ được khởi tạo trong landmarks.py)
CHOSEN_LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
CHOSEN_RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

# ===================================================================
# PREPROCESSING PARAMETERS
# ===================================================================
SEED = 42
TEST_SIZE = 0.20
BATCH_SIZE = 64

# ===================================================================
# FACE DETECTION PARAMETERS
# ===================================================================
FACE_DETECTION_SCALE_FACTOR = 1.3
FACE_DETECTION_MIN_NEIGHBORS = 5

# ===================================================================
# MEDIAPIPE PARAMETERS
# ===================================================================
MEDIAPIPE_CONFIG = {
    'static_image_mode': True,
    'max_num_faces': 1,
    'refine_landmarks': False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# ===================================================================
# LOGGING
# ===================================================================
VERBOSE = True
PROGRESS_INTERVAL = 500  # Print progress every N images