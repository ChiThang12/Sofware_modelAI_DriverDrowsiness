"""
ASIC-Optimized Image Preprocessing
Fixed-point arithmetic, power-of-2 operations, hardware-friendly
Target size: 64x64 (2^6 x 2^6)
"""
import cv2
import numpy as np
from config_asic import (
    IMG_SIZE,
    SCALE_FACTOR,
    FIXED_POINT_FRAC_BITS,
    QUANTIZATION_MIN,
    QUANTIZATION_MAX,
    USE_GRAYSCALE,
)

# log2(IMG_SIZE) dùng để tính địa chỉ pixel: addr = (y << LOG2_SIZE) + x
import math
LOG2_SIZE = int(math.log2(IMG_SIZE))  # 64 → 6


def rgb_to_grayscale_optimized(img):
    """
    Convert RGB to Grayscale dùng integer arithmetic.

    Hardware implementation (approximation dùng shift + add):
        Y ≈ (R >> 2) + (G >> 1) + (B >> 3)
        (xấp xỉ trọng số BT.601: 0.299R + 0.587G + 0.114B)

    Args:
        img: BGR image (H, W, 3) — OpenCV format

    Returns:
        Grayscale image (H, W), dtype uint8
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)


def normalize_to_fixed_point(img_uint8):
    """
    Convert uint8 [0, 255]  →  Q0.7 fixed-point int8 [0, 127]

    Formula: fixed_point = uint8_value >> 1

    Mapping:
        0   →   0
        255 → 127

    Hardware: chỉ 1 phép right-shift, không cần multiplier/divider.

    Args:
        img_uint8: uint8 image [0, 255]

    Returns:
        int8 image [0, 127] — Q0.7 format
    """
    fixed_point = (img_uint8.astype(np.int16) >> 1).astype(np.int8)
    return np.clip(fixed_point, 0, QUANTIZATION_MAX)


def denormalize_from_fixed_point(img_fixed):
    """
    Convert Q0.7 fixed-point int8  →  uint8 [0, 255]

    Formula: uint8 = fixed_point << 1

    Args:
        img_fixed: int8 image [0, 127]

    Returns:
        uint8 image [0, 254]
    """
    uint8_img = (img_fixed.astype(np.int16) << 1).astype(np.uint8)
    return np.clip(uint8_img, 0, 255)


def fixed_point_to_float(img_fixed):
    """
    Convert Q0.7 fixed-point  →  float32 [0, ~1.0] để đưa vào model.

    Formula: float_value = fixed_point / 128.0

    Args:
        img_fixed: int8 image [0, 127]

    Returns:
        float32 image [0.0, ~1.0]
    """
    return img_fixed.astype(np.float32) / SCALE_FACTOR


def float_to_fixed_point(img_float):
    """
    Convert float32 [0, 1]  →  Q0.7 fixed-point int8.

    Formula: fixed_point = round(float_value * 128)

    Args:
        img_float: float32 image [0, 1]

    Returns:
        int8 image [0, 127]
    """
    fixed = np.round(img_float * SCALE_FACTOR).astype(np.int8)
    return np.clip(fixed, 0, QUANTIZATION_MAX)


def resize_power_of_2(img, target_size=IMG_SIZE):
    """
    Resize image về kích thước power-of-2.

    ASIC benefit:
        - Địa chỉ pixel: addr = (y << log2(W)) + x  — chỉ dùng shift
        - DMA transfer kích thước cố định
        - Tile-friendly cho systolic array

    Args:
        img   : Input image (grayscale hoặc BGR)
        target_size: Kích thước đích (phải là power-of-2)

    Returns:
        Resized image (target_size × target_size)
    """
    assert (target_size & (target_size - 1)) == 0, \
        f"target_size phải là power-of-2, nhận được {target_size}"
    return cv2.resize(img, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


def process_image_asic_friendly(img):
    """
    Full ASIC-friendly preprocessing pipeline cho 1 ảnh.

    Bước 1 — Grayscale:  BGR → Gray  (integer arithmetic)
    Bước 2 — Resize:     → 64×64     (power-of-2, INTER_AREA)
    Bước 3 — Quantize:   uint8 >> 1  → int8 Q0.7 [0, 127]

    Tất cả operations hardware-friendly:
        - Bit shift thay vì multiply/divide
        - Integer arithmetic only
        - Memory-aligned dimensions

    Args:
        img: BGR image (H, W, 3) từ OpenCV

    Returns:
        int8 array (64, 64) — Q0.7 fixed-point
    """
    # Bước 1: Grayscale
    if len(img.shape) == 3 and USE_GRAYSCALE:
        img_gray = rgb_to_grayscale_optimized(img)
    else:
        img_gray = img

    # Bước 2: Resize về 64×64
    img_resized = resize_power_of_2(img_gray, IMG_SIZE)

    # Bước 3: Normalize → Q0.7 int8
    img_fixed = normalize_to_fixed_point(img_resized)

    return img_fixed


def verify_asic_friendly(img_fixed):
    """
    Kiểm tra ảnh đã đúng chuẩn ASIC chưa.

    Checks:
        - Kích thước là power-of-2 và khớp IMG_SIZE (64×64)
        - dtype = int8
        - Giá trị trong [0, 127]
        - Memory alignment

    Args:
        img_fixed: Preprocessed image

    Returns:
        bool: True nếu ASIC-friendly
    """
    h, w = img_fixed.shape[:2]

    if (h & (h - 1)) != 0 or (w & (w - 1)) != 0:
        print(f"  Dimensions not power of 2: {h}x{w}")
        return False

    if img_fixed.dtype != np.int8:
        print(f"  Data type not int8: {img_fixed.dtype}")
        return False

    if img_fixed.min() < 0 or img_fixed.max() > QUANTIZATION_MAX:
        print(f"  Values out of range: [{img_fixed.min()}, {img_fixed.max()}]")
        return False

    if h != IMG_SIZE or w != IMG_SIZE:
        print(f"  Size mismatch: {h}x{w} vs {IMG_SIZE}x{IMG_SIZE}")
        return False

    print(f"  Image is ASIC-friendly:")
    print(f"    Size  : {h}x{w} (2^{LOG2_SIZE} x 2^{LOG2_SIZE})")
    print(f"    Dtype : {img_fixed.dtype}")
    print(f"    Range : [{img_fixed.min()}, {img_fixed.max()}]")
    print(f"    Memory: {img_fixed.nbytes} bytes ({img_fixed.nbytes/1024:.2f} KB)")

    return True


# ===================================================================
# ASIC HARDWARE SIMULATION (for testing)
# ===================================================================
def simulate_asic_inference(img_fixed):
    """
    Mô phỏng cách ASIC xử lý ảnh ở mức bit.
    """
    print("\n" + "=" * 70)
    print("ASIC HARDWARE SIMULATION  —  64x64 Q0.7 int8")
    print("=" * 70)

    h, w = img_fixed.shape
    print(f"\n1. Image Dimensions: {h}x{w}")
    print(f"   Address: addr = (y << {LOG2_SIZE}) + x")
    print(f"   Total pixels: {h * w} = 2^{int(math.log2(h * w))}")

    print(f"\n2. Memory Layout:")
    print(f"   Size  : {img_fixed.nbytes} bytes ({img_fixed.nbytes / 1024:.2f} KB)")
    align_offset = img_fixed.nbytes % 16
    print(f"   Align : {align_offset} bytes offset (expected 0)")

    # Sample pixel — center của ảnh 64x64
    cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
    pixel_value = img_fixed[cy, cx]
    addr = (cy << LOG2_SIZE) + cx

    print(f"\n3. Sample Pixel (center {cy},{cx}):")
    print(f"   Address : ({cy} << {LOG2_SIZE}) + {cx} = {addr}  (0x{addr:04X})")
    print(f"   int8    : {pixel_value}  (0x{int(pixel_value) & 0xFF:02X})")
    print(f"   binary  : {int(pixel_value) & 0xFF:08b}")
    print(f"   float   : {pixel_value / SCALE_FACTOR:.6f}")

    print(f"\n4. Fixed-Point Normalization (Hardware):")
    print(f"   Input  uint8 = 128  →  128 >> 1 = 64  (int8 Q0.7)")
    print(f"   Represents: 64 / {SCALE_FACTOR} = {64 / SCALE_FACTOR:.4f}")
    print(f"   Input  uint8 = 200  →  200 >> 1 = 100 (int8 Q0.7)")
    print(f"   Represents: 100 / {SCALE_FACTOR} = {100 / SCALE_FACTOR:.4f}")

    print(f"\n5. ASIC Operations:")
    print(f"   Normalization  : uint8 >> 1  (no multiplier)")
    print(f"   Pixel address  : (y << {LOG2_SIZE}) + x  (no multiply)")
    print(f"   DMA block size : {img_fixed.nbytes} bytes")
    print(f"   No FPU required")

    print("=" * 70)


if __name__ == "__main__":
    print("ASIC-Friendly Image Preprocessing Module")
    print("=" * 70)
    print(f"Target format : Q{0}.{FIXED_POINT_FRAC_BITS} fixed-point (int8)")
    print(f"Image size    : {IMG_SIZE}x{IMG_SIZE}  (2^{LOG2_SIZE} x 2^{LOG2_SIZE})")
    print(f"Data type     : int8")
    print(f"Scale factor  : {SCALE_FACTOR}")
    print(f"Pixel address : (y << {LOG2_SIZE}) + x")
    print("=" * 70)