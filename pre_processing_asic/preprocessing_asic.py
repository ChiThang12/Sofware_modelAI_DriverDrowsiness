"""
ASIC-Optimized Image Preprocessing
Fixed-point arithmetic, power-of-2 operations, hardware-friendly
"""
import cv2
import numpy as np
from config_asic import (
    IMG_SIZE,
    SCALE_FACTOR,
    FIXED_POINT_FRAC_BITS,
    QUANTIZATION_MIN,
    QUANTIZATION_MAX,
    USE_GRAYSCALE
)


def rgb_to_grayscale_optimized(img):
    """
    Convert RGB to Grayscale using integer arithmetic
    Uses approximation: Y = (R + G + B) / 3
    
    Hardware implementation:
    Y = (R + G + B) >> 2  (approximate, very fast)
    or
    Y = ((R << 1) + (R << 2) + G + (G << 2) + (G << 3) + B) >> 4
       (more accurate, uses shifts and adds only)
    
    Args:
        img: RGB image (H, W, 3)
    
    Returns:
        Grayscale image (H, W)
    """
    # Option 1: OpenCV (for preprocessing, will be replaced in ASIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Option 2: Simple average (ASIC-friendly)
    # gray = (img[:, :, 0].astype(np.uint16) + 
    #         img[:, :, 1].astype(np.uint16) + 
    #         img[:, :, 2].astype(np.uint16)) // 3
    
    # Option 3: Weighted (more accurate, ASIC implementation uses shifts)
    # Y = 0.299*R + 0.587*G + 0.114*B
    # Approximation using shifts: Y ≈ (R>>2) + (G>>1) + (B>>3)
    
    return gray.astype(np.uint8)


def normalize_to_fixed_point(img_uint8):
    """
    Convert uint8 [0, 255] to Q0.7 fixed-point int8 [0, 127]
    
    Formula: fixed_point = (uint8_value >> 1)
    
    This maps:
    - 0 → 0
    - 255 → 127
    
    In hardware: Just 1-bit right shift!
    
    Args:
        img_uint8: uint8 image [0, 255]
    
    Returns:
        int8 image [0, 127] in Q0.7 format
    """
    # Method 1: Direct right shift (hardware uses this)
    # fixed_point = img_uint8 >> 1
    
    # Method 2: Scale and convert (mathematically equivalent)
    # Range [0, 255] → [0, 127]
    fixed_point = (img_uint8.astype(np.int16) >> 1).astype(np.int8)
    
    # Ensure values are in valid range
    fixed_point = np.clip(fixed_point, 0, QUANTIZATION_MAX)
    
    return fixed_point


def denormalize_from_fixed_point(img_fixed):
    """
    Convert Q0.7 fixed-point int8 back to uint8
    
    Formula: uint8 = (fixed_point << 1)
    
    Args:
        img_fixed: int8 image [0, 127] in Q0.7
    
    Returns:
        uint8 image [0, 255]
    """
    # Left shift by 1 bit
    uint8_img = (img_fixed.astype(np.int16) << 1).astype(np.uint8)
    return np.clip(uint8_img, 0, 255)


def fixed_point_to_float(img_fixed):
    """
    Convert Q0.7 fixed-point to float [0, 1] for training
    
    Formula: float_value = fixed_point / 128.0
    
    Args:
        img_fixed: int8 image [0, 127]
    
    Returns:
        float32 image [0, ~1.0]
    """
    return img_fixed.astype(np.float32) / SCALE_FACTOR


def float_to_fixed_point(img_float):
    """
    Convert float [0, 1] to Q0.7 fixed-point
    
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
    Resize image to power-of-2 dimensions
    
    ASIC benefit: Power-of-2 size enables:
    - Simple addressing: addr = (y << log2(width)) + x
    - Efficient tiling
    - Easy DMA transfers
    
    Args:
        img: Input image
        target_size: Target size (must be power of 2)
    
    Returns:
        Resized image
    """
    # Verify power of 2
    assert (target_size & (target_size - 1)) == 0, "Size must be power of 2"
    
    return cv2.resize(img, (target_size, target_size), 
                     interpolation=cv2.INTER_AREA)


def process_image_asic_friendly(img):
    """
    Full ASIC-friendly preprocessing pipeline
    
    Steps:
    1. Convert to grayscale (if needed)
    2. Resize to power-of-2 (128x128)
    3. Normalize to Q0.7 fixed-point [0, 127]
    
    All operations are hardware-friendly:
    - Bit shifts instead of multiply/divide
    - Integer arithmetic only
    - Memory-aligned dimensions
    
    Args:
        img: Input image (BGR format from OpenCV)
    
    Returns:
        Preprocessed image in Q0.7 fixed-point format (int8)
    """
    # Step 1: Grayscale conversion
    if len(img.shape) == 3 and USE_GRAYSCALE:
        img_gray = rgb_to_grayscale_optimized(img)
    else:
        img_gray = img
    
    # Step 2: Resize to power-of-2
    img_resized = resize_power_of_2(img_gray, IMG_SIZE)
    
    # Step 3: Convert to Q0.7 fixed-point
    img_fixed = normalize_to_fixed_point(img_resized)
    
    return img_fixed


def verify_asic_friendly(img_fixed):
    """
    Verify that image is ASIC-friendly
    
    Checks:
    - Dimensions are power of 2
    - Data type is int8
    - Values in valid range
    - Memory alignment
    
    Args:
        img_fixed: Preprocessed image
    
    Returns:
        bool: True if ASIC-friendly
    """
    h, w = img_fixed.shape[:2]
    
    # Check power of 2
    if (h & (h - 1)) != 0 or (w & (w - 1)) != 0:
        print(f"❌ Dimensions not power of 2: {h}x{w}")
        return False
    
    # Check dtype
    if img_fixed.dtype != np.int8:
        print(f"❌ Data type not int8: {img_fixed.dtype}")
        return False
    
    # Check range
    if img_fixed.min() < 0 or img_fixed.max() > QUANTIZATION_MAX:
        print(f"❌ Values out of range: [{img_fixed.min()}, {img_fixed.max()}]")
        return False
    
    # Check size matches config
    if h != IMG_SIZE or w != IMG_SIZE:
        print(f"❌ Size mismatch: {h}x{w} vs {IMG_SIZE}x{IMG_SIZE}")
        return False
    
    print(f"✅ Image is ASIC-friendly:")
    print(f"   Size: {h}x{w} (power of 2)")
    print(f"   Dtype: {img_fixed.dtype}")
    print(f"   Range: [{img_fixed.min()}, {img_fixed.max()}]")
    print(f"   Memory: {img_fixed.nbytes} bytes")
    
    return True


# ===================================================================
# ASIC HARDWARE SIMULATION (for testing)
# ===================================================================
def simulate_asic_inference(img_fixed):
    """
    Simulate how ASIC would process the image
    
    This shows the bit-level operations ASIC will perform
    """
    print("\n" + "="*70)
    print("ASIC HARDWARE SIMULATION")
    print("="*70)
    
    # Image dimensions
    h, w = img_fixed.shape
    print(f"\n1. Image Dimensions: {h}x{w}")
    print(f"   Address calculation: addr = (y << 7) + x")
    print(f"   Total pixels: {h * w} = 2^14")
    
    # Memory layout
    print(f"\n2. Memory Layout:")
    print(f"   Base address: 0x0000")
    print(f"   Size: {img_fixed.nbytes} bytes ({img_fixed.nbytes/1024:.1f} KB)")
    print(f"   Alignment: {img_fixed.nbytes % 16} bytes offset (should be 0)")
    
    # Sample pixel access
    y, x = 64, 64  # Center pixel
    pixel_value = img_fixed[y, x]
    addr = (y << 7) + x  # Bit-shift addressing
    
    print(f"\n3. Sample Pixel Access:")
    print(f"   Coordinates: ({y}, {x})")
    print(f"   Address: (64 << 7) + 64 = {addr} (0x{addr:04X})")
    print(f"   Value: {pixel_value} (0x{pixel_value:02X})")
    print(f"   Binary: {pixel_value:08b}")
    print(f"   Float: {pixel_value / SCALE_FACTOR:.6f}")
    
    # Demonstrate bit-shift normalization
    print(f"\n4. Fixed-Point Conversion (Hardware):")
    print(f"   Input uint8: 128 (example)")
    print(f"   Right shift 1: 128 >> 1 = 64")
    print(f"   Output int8: 64 (Q0.7 format)")
    print(f"   Represents: 64/128 = 0.5 (float)")
    
    print(f"\n5. ASIC Operations:")
    print(f"   ✓ No multiplication for normalization (just bit shift)")
    print(f"   ✓ No division (right shift = divide by 2^n)")
    print(f"   ✓ No floating-point unit needed")
    print(f"   ✓ Simple indexing with bit shifts")
    print(f"   ✓ DMA-friendly contiguous memory")
    
    print("="*70)


if __name__ == "__main__":
    print("ASIC-Friendly Image Preprocessing Module")
    print("="*70)
    print(f"Target format: Q0.{FIXED_POINT_FRAC_BITS} fixed-point")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} (2^7 x 2^7)")
    print(f"Data type: int8")
    print(f"Scale factor: {SCALE_FACTOR}")
    print("="*70)