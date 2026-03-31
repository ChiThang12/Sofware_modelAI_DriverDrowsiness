"""
ASIC-Optimized Configuration for Driver Drowsiness Detection
Fixed-point arithmetic, power-of-2 operations, minimal precision
Target: 64x64 input — đủ rõ vùng mắt (~8-12px), tiết kiệm 4x memory so với 128x128
"""
import os

# ===================================================================
# PATH CONFIGURATION
# ===================================================================
BASE_DIR    = r"D:\PROJECTDriverDrowsiness"
DATASET_DIR = os.path.join(BASE_DIR, "FaceImages")
ACTIVE_DIR  = os.path.join(DATASET_DIR, "ActiveSubjects")
FATIGUE_DIR = os.path.join(DATASET_DIR, "FatigueSubjects")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "ModelAiFinal/PreprocessedData_ASIC")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_TRAIN_FILE = os.path.join(OUTPUT_DIR, "X_train_asic.npy")
X_TEST_FILE  = os.path.join(OUTPUT_DIR, "X_test_asic.npy")
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "y_train_asic.npy")
Y_TEST_FILE  = os.path.join(OUTPUT_DIR, "y_test_asic.npy")

# Cache files
X_CACHE_FILE = os.path.join(OUTPUT_DIR, "X_preprocessed_asic.npy")
Y_CACHE_FILE = os.path.join(OUTPUT_DIR, "y_preprocessed_asic.npy")

# ===================================================================
# ASIC-OPTIMIZED IMAGE CONFIGURATION
# ===================================================================
# 64x64 = 2^6 x 2^6  (power-of-2)
# Lý do chọn 64x64:
#   - Vùng mắt ~8-12px: đủ để model phân biệt mở/nhắm
#   - Memory/frame: 4KB (tiết kiệm 4x so với 128x128 = 16KB)
#   - Địa chỉ pixel: addr = (y << 6) + x  (đơn giản hơn << 7)
#   - Param budget: DWConv + GAP chỉ ~15-25k params, dư cho 50k limit
IMG_SIZE = 64

CATEGORIES = ["FatigueSubjects", "ActiveSubjects"]

# ===================================================================
# ASIC FIXED-POINT CONFIGURATION
# ===================================================================
USE_GRAYSCALE   = True
USE_FIXED_POINT = True

# Q0.7 fixed-point (int8)
# Range lưu trữ: [0, 127]
# Chuyển về float khi train: value / 128.0  →  [0.0, ~1.0]
# Hardware normalize: uint8 >> 1  (chỉ 1 phép shift)
FIXED_POINT_INT_BITS   = 0
FIXED_POINT_FRAC_BITS  = 7
FIXED_POINT_TOTAL_BITS = 8

QUANTIZATION_BITS = 8
QUANTIZATION_MIN  = -128
QUANTIZATION_MAX  = 127

# 128 = 2^7 — hardware dùng right-shift thay vì divide
SCALE_FACTOR = 2 ** FIXED_POINT_FRAC_BITS  # 128

# ===================================================================
# ASIC HARDWARE CONSTRAINTS
# ===================================================================
MEMORY_ALIGNMENT = 16   # bytes — căn chỉnh cho DMA transfer
BATCH_SIZE       = 1    # ASIC xử lý 1 ảnh/lần
DATA_WIDTH       = 8    # bits per pixel

# ===================================================================
# IMAGE PREPROCESSING PIPELINE
# ===================================================================
PREPROCESSING_STEPS = {
    'face_detection'     : True,
    'mediapipe_landmarks': True,
    'resize'             : True,
    'grayscale'          : True,
    'normalization'      : 'fixed_point',  # Q0.7
    'clipping'           : True,
}

# Hardware-friendly: x / 128 = x >> 1  (right shift 1 bit)
NORM_METHOD = 'div128'

# ===================================================================
# FACE DETECTION PARAMETERS
# ===================================================================
FACE_DETECTION_SCALE_FACTOR  = 1.3
FACE_DETECTION_MIN_NEIGHBORS = 5

# ===================================================================
# MEDIAPIPE PARAMETERS
# ===================================================================
MEDIAPIPE_CONFIG = {
    'static_image_mode'       : True,
    'max_num_faces'           : 1,
    'refine_landmarks'        : False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence' : 0.5,
}

# ===================================================================
# TRAIN/TEST SPLIT
# ===================================================================
SEED      = 42
TEST_SIZE = 0.20

# ===================================================================
# ASIC-SPECIFIC OPTIMIZATIONS
# ===================================================================
ASIC_OPTIMIZATIONS = {
    'power_of_2_size'       : True,
    'use_bit_shifts'        : True,
    'memory_aligned'        : True,
    'fixed_point_arithmetic': True,
    'quantization_aware'    : True,
    'pipeline_friendly'     : True,
}

# ===================================================================
# LOGGING
# ===================================================================
VERBOSE           = True
PROGRESS_INTERVAL = 500

# ===================================================================
# HARDWARE SPECIFICATIONS (for reference)
# ===================================================================
ASIC_SPECS = {
    'name'            : 'Driver Drowsiness Detection ASIC',
    'process_node'    : '28nm',
    'max_frequency'   : '200MHz',
    'data_width'      : 8,
    'mac_units'       : 64,
    'memory_bandwidth': '10GB/s',
    'power_budget'    : '100mW',
}

print(f"""
╔════════════════════════════════════════════════════════════════╗
║           ASIC-OPTIMIZED CONFIGURATION LOADED                  ║
╚════════════════════════════════════════════════════════════════╝

Image Size:      {IMG_SIZE}x{IMG_SIZE}  (2^6 x 2^6, power-of-2)
Data Format:     Q{FIXED_POINT_INT_BITS}.{FIXED_POINT_FRAC_BITS} fixed-point (int8)
Quantization:    {QUANTIZATION_BITS}-bit
Scale Factor:    {SCALE_FACTOR} (2^{FIXED_POINT_FRAC_BITS})
Normalization:   uint8 >> 1  (hardware: 1-bit right shift)

Memory/Image:    {IMG_SIZE * IMG_SIZE} bytes ({IMG_SIZE * IMG_SIZE / 1024:.2f} KB)
Alignment:       {MEMORY_ALIGNMENT} bytes
Pixel address:   addr = (y << 6) + x

ASIC Benefits:
  ✓ Bit-shift normalization (uint8 >> 1, không cần multiplier)
  ✓ Power-of-2 addressing (y << 6 + x)
  ✓ No FPU required
  ✓ 4x ít memory hơn 128x128 (4KB vs 16KB/frame)
  ✓ Vùng mắt ~8-12px — đủ phân biệt mở/nhắm
  ✓ Param budget ~15-25k với DWConv+GAP, dư cho 50k limit
""")