"""
Master script — ASIC-optimized preprocessing pipeline
Chạy đủ 3 bước: preprocess → split → verify
Output: X_train/test (N, 64, 64) int8 Q0.7
"""
import os
import sys
import time
import math
from datetime import datetime
from config_asic import OUTPUT_DIR, IMG_SIZE, FIXED_POINT_FRAC_BITS

LOG2_SIZE = int(math.log2(IMG_SIZE))  # 64 → 6


# ===================================================================
# CHECKS
# ===================================================================
def check_dependencies():
    print("  Checking dependencies...")
    required = ['cv2', 'numpy', 'mediapipe', 'sklearn', 'tqdm']
    missing  = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"    {pkg}")
        except ImportError:
            print(f"    MISSING: {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\n  Missing: {', '.join(missing)}")
        print(f"  Install: pip install {' '.join(missing)}")
        print(f"  Note: cv2 → pip install opencv-python")
        return False

    print("  All dependencies OK\n")
    return True


def check_dataset():
    from config_asic import ACTIVE_DIR, FATIGUE_DIR
    print("  Checking dataset...")

    for name, path in [("Active", ACTIVE_DIR), ("Fatigue", FATIGUE_DIR)]:
        if not os.path.exists(path):
            print(f"  NOT FOUND: {path}")
            return False
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        print(f"    {name}: {count} images")

    return True


def check_config_size():
    """Đảm bảo IMG_SIZE trong config đúng là 64."""
    if IMG_SIZE != 64:
        print(f"\n  WARNING: IMG_SIZE = {IMG_SIZE}, expected 64.")
        print(f"  Hãy sửa config_asic.py: IMG_SIZE = 64")
        return False
    print(f"  IMG_SIZE = {IMG_SIZE}  (2^{LOG2_SIZE})  OK")
    return True


# ===================================================================
# DISPLAY CONFIG
# ===================================================================
def display_asic_config():
    from config_asic import (
        IMG_SIZE, FIXED_POINT_FRAC_BITS, SCALE_FACTOR,
        QUANTIZATION_BITS, USE_FIXED_POINT,
    )
    print("=" * 70)
    print("ASIC CONFIGURATION")
    print("=" * 70)
    print(f"  Image Size     : {IMG_SIZE}x{IMG_SIZE}  (2^{LOG2_SIZE} x 2^{LOG2_SIZE})")
    print(f"  Fixed-Point    : Q0.{FIXED_POINT_FRAC_BITS}  (int8)")
    print(f"  Quantization   : {QUANTIZATION_BITS}-bit")
    print(f"  Scale Factor   : {SCALE_FACTOR}  (2^{FIXED_POINT_FRAC_BITS})")
    print(f"  Use Fixed-Point: {USE_FIXED_POINT}")
    print(f"\n  Hardware Benefits:")
    print(f"    Normalize     : uint8 >> 1  (no multiplier)")
    print(f"    Pixel address : (y << {LOG2_SIZE}) + x")
    print(f"    No FPU needed")
    print(f"    Memory/frame  : {IMG_SIZE * IMG_SIZE} bytes  ({IMG_SIZE * IMG_SIZE / 1024:.2f} KB)")
    print(f"    4x less than 128x128  (4KB vs 16KB)")
    print("=" * 70 + "\n")


# ===================================================================
# PIPELINE
# ===================================================================
def run_asic_preprocessing():
    print("=" * 70)
    print("ASIC DRIVER DROWSINESS — FULL PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start = time.time()

    # ── Pre-checks ────────────────────────────────────────────────
    if not check_dependencies():
        return False
    if not check_config_size():
        return False
    if not check_dataset():
        return False

    display_asic_config()

    # ── STEP 1: Preprocess ────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 — ASIC-OPTIMIZED PREPROCESSING")
    print("=" * 70)
    from preprocess_data_asic import load_and_preprocess_asic

    try:
        X, y = load_and_preprocess_asic(use_cache=True)

        # Kiểm tra shape output đúng 64x64
        assert X.shape[1] == IMG_SIZE and X.shape[2] == IMG_SIZE, \
            f"Shape mismatch: {X.shape}, expected (N,{IMG_SIZE},{IMG_SIZE})"
        assert X.dtype == np.int8, f"dtype mismatch: {X.dtype}"

        print(f"\n  Step 1 OK: {len(X)} images")
        print(f"    Shape : {X.shape}")
        print(f"    Format: Q0.7 fixed-point (int8)")
    except Exception as e:
        print(f"\n  Step 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── STEP 2: Split ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — TRAIN/TEST SPLIT")
    print("=" * 70)
    from split_data_asic import split_and_save_asic

    try:
        X_train, X_test, y_train, y_test = split_and_save_asic()
        print(f"\n  Step 2 OK")
    except Exception as e:
        print(f"\n  Step 2 FAILED: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── STEP 3: Verify ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — VERIFY ASIC FORMAT")
    print("=" * 70)
    from preprocessing_asic import verify_asic_friendly, simulate_asic_inference

    try:
        print("\n  Verifying X_train[0]...")
        ok = verify_asic_friendly(X_train[0])
        if not ok:
            raise ValueError("verify_asic_friendly failed")

        print("\n  ASIC hardware simulation:")
        simulate_asic_inference(X_train[0])
        print(f"\n  Step 3 OK")
    except Exception as e:
        print(f"\n  Step 3 FAILED: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print(f"  Total time : {elapsed / 60:.2f} min")
    print(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n  Output → {OUTPUT_DIR}")
    print(f"\n  Files ready for Google Colab:")
    print(f"    X_train_asic.npy  {X_train.shape}")
    print(f"    X_test_asic.npy   {X_test.shape}")
    print(f"    y_train_asic.npy  {y_train.shape}")
    print(f"    y_test_asic.npy   {y_test.shape}")

    print(f"\n  ASIC Format:")
    print(f"    Q0.{FIXED_POINT_FRAC_BITS} fixed-point (int8)")
    print(f"    {IMG_SIZE}x{IMG_SIZE}  (2^{LOG2_SIZE} x 2^{LOG2_SIZE})")
    print(f"    Range: [0, 127]")
    print(f"    Frame size: {X_train[0].nbytes} bytes")

    print(f"\n  Convert to float khi train:")
    print(f"    X_float = X.astype(np.float32) / 128.0")

    print(f"\n  Performance @ 200 MHz:")
    cycles = IMG_SIZE * IMG_SIZE
    lat_us = cycles / 200e6 * 1e6
    fps    = 1e6 / lat_us
    print(f"    Latency    : ~{lat_us:.1f} µs/frame")
    print(f"    Throughput : ~{fps:.0f} FPS  (simplified estimate)")

    return True


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    import numpy as np   # cần cho assert trong pipeline

    success = run_asic_preprocessing()

    if success:
        print("\n  All steps completed successfully.")
        print("\n  Next steps:")
        print("    1. Upload .npy files lên Google Drive")
        print("    2. Train TinyCNN (DWConv + GAP, ~20k params) trên Colab")
        print("    3. Quantization-aware training (QAT) với TFLite INT8")
        print("    4. Implement RTL / synthesize ASIC")
        sys.exit(0)
    else:
        print("\n  Pipeline failed. Check errors above.")
        sys.exit(1)