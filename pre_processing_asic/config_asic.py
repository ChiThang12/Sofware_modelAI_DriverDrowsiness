"""
ASIC-Optimized Configuration for Driver Drowsiness Detection
Fixed-point arithmetic, power-of-2 operations, minimal precision
"""
import os

# ===================================================================
# PATH CONFIGURATION
# ===================================================================
BASE_DIR = r"D:\PROJECTDriverDrowsiness"
DATASET_DIR = os.path.join(BASE_DIR, "FaceImages")
ACTIVE_DIR = os.path.join(DATASET_DIR, "ActiveSubjects")
FATIGUE_DIR = os.path.join(DATASET_DIR, "FatigueSubjects")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "ModelAiFinal/PreprocessedData_ASIC")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_TRAIN_FILE = os.path.join(OUTPUT_DIR, "X_train_asic.npy")
X_TEST_FILE = os.path.join(OUTPUT_DIR, "X_test_asic.npy")
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "y_train_asic.npy")
Y_TEST_FILE = os.path.join(OUTPUT_DIR, "y_test_asic.npy")

# Cache files
X_CACHE_FILE = os.path.join(OUTPUT_DIR, "X_preprocessed_asic.npy")
Y_CACHE_FILE = os.path.join(OUTPUT_DIR, "y_preprocessed_asic.npy")

# ===================================================================
# ASIC-OPTIMIZED IMAGE CONFIGURATION
# ===================================================================
# Power-of-2 image size for efficient memory access
IMG_SIZE = 128  # Changed from 145 to 128 (2^7)
                # Hardware can use simple bit-shift addressing

CATEGORIES = ["FatigueSubjects", "ActiveSubjects"]

# ===================================================================
# ASIC FIXED-POINT CONFIGURATION
# ===================================================================
USE_GRAYSCALE = True
USE_FIXED_POINT = True

# Fixed-point precision
# Format: Qm.n where m = integer bits, n = fractional bits
# Total bits = m + n + 1 (sign bit)
FIXED_POINT_INT_BITS = 0      # No integer part needed for [0,1] range
FIXED_POINT_FRAC_BITS = 7     # 7 fractional bits
FIXED_POINT_TOTAL_BITS = 8    # int8: 1 sign + 0 int + 7 frac

# For int8: Q0.7 format
# Range: [-1, 0.9921875] with step size 1/128
# Perfect for normalized image data [0, 1]

# Alternative formats (comment/uncomment as needed):
# Q1.6: 1 int bit, 6 frac bits - range [-2, 1.984375]
# Q0.7: 0 int bit, 7 frac bits - range [-1, 0.9921875] ← RECOMMENDED

# Quantization settings
QUANTIZATION_BITS = 8         # 8-bit quantization (int8)
QUANTIZATION_MIN = -128       # int8 min
QUANTIZATION_MAX = 127        # int8 max

# Scaling factor for Q0.7: 128 = 2^7
SCALE_FACTOR = 2 ** FIXED_POINT_FRAC_BITS  # 128

# ===================================================================
# ASIC HARDWARE CONSTRAINTS
# ===================================================================
# Memory alignment for efficient DMA transfer
MEMORY_ALIGNMENT = 16  # bytes (typical for ASIC)

# Batch size (power of 2 for hardware efficiency)
BATCH_SIZE = 1  # ASIC typically processes 1 image at a time

# Data width (bits per pixel)
DATA_WIDTH = 8  # 8-bit per pixel

# ===================================================================
# IMAGE PREPROCESSING PIPELINE FOR ASIC
# ===================================================================
PREPROCESSING_STEPS = {
    'face_detection': True,
    'mediapipe_landmarks': True,  # Keep for accuracy
    'resize': True,
    'grayscale': True,
    'normalization': 'fixed_point',  # Q0.7 fixed-point
    'clipping': True,  # Clip values to prevent overflow
}

# Normalization method
NORM_METHOD = 'div128'  # Divide by 128 (right shift 7 bits)
                        # Hardware-friendly: x >> 7

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
# TRAIN/TEST SPLIT
# ===================================================================
SEED = 42
TEST_SIZE = 0.20

# ===================================================================
# ASIC-SPECIFIC OPTIMIZATIONS
# ===================================================================
ASIC_OPTIMIZATIONS = {
    # Use power-of-2 dimensions
    'power_of_2_size': True,
    
    # Use bit-shift operations instead of multiply/divide
    'use_bit_shifts': True,
    
    # Memory alignment for DMA
    'memory_aligned': True,
    
    # Fixed-point arithmetic
    'fixed_point_arithmetic': True,
    
    # Quantization-aware
    'quantization_aware': True,
    
    # Pipeline-friendly (sequential processing)
    'pipeline_friendly': True,
}

# ===================================================================
# LOGGING
# ===================================================================
VERBOSE = True
PROGRESS_INTERVAL = 500

# ===================================================================
# HARDWARE SPECIFICATIONS (for reference)
# ===================================================================
ASIC_SPECS = {
    'name': 'Driver Drowsiness Detection ASIC',
    'process_node': '28nm',  # Example
    'max_frequency': '200MHz',
    'data_width': 8,  # 8-bit datapath
    'mac_units': 64,  # Number of MAC units
    'memory_bandwidth': '10GB/s',
    'power_budget': '100mW',
}

print(f"""
╔════════════════════════════════════════════════════════════════╗
║           ASIC-OPTIMIZED CONFIGURATION LOADED                  ║
╚════════════════════════════════════════════════════════════════╝

Image Size:      {IMG_SIZE}x{IMG_SIZE} (power-of-2)
Data Format:     Q{FIXED_POINT_INT_BITS}.{FIXED_POINT_FRAC_BITS} fixed-point (int8)
Quantization:    {QUANTIZATION_BITS}-bit
Scale Factor:    {SCALE_FACTOR} (2^{FIXED_POINT_FRAC_BITS})
Normalization:   x / {SCALE_FACTOR} = x >> {FIXED_POINT_FRAC_BITS}

Memory/Image:    {IMG_SIZE * IMG_SIZE} bytes ({IMG_SIZE * IMG_SIZE / 1024:.1f} KB)
Alignment:       {MEMORY_ALIGNMENT} bytes

ASIC Benefits:
  ✓ Bit-shift operations (no multipliers for normalization)
  ✓ Power-of-2 addressing (simple indexing)
  ✓ Fixed-point arithmetic (no FPU needed)
  ✓ Minimal memory footprint
  ✓ Pipeline-friendly sequential processing
""")