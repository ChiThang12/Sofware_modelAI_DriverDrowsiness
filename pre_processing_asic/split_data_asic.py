"""
Train/Test split cho ASIC-optimized data
Input: X_preprocessed_asic.npy  — shape (N, 64, 64), dtype int8
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split

from config_asic import (
    X_CACHE_FILE,
    Y_CACHE_FILE,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    SEED,
    TEST_SIZE,
    CATEGORIES,
    IMG_SIZE,
    FIXED_POINT_FRAC_BITS,
    SCALE_FACTOR,
)
import math
LOG2_SIZE = int(math.log2(IMG_SIZE))  # 64 → 6


def split_and_save_asic():
    """
    Load ASIC preprocessed data, split train/test, lưu file.

    Returns:
        X_train, X_test  : int8 array (N, 64, 64)
        y_train, y_test  : int32 array (N,)
    """
    print("\n  Splitting ASIC data...")

    # ── Load ──────────────────────────────────────────────────────
    if not (os.path.exists(X_CACHE_FILE) and os.path.exists(Y_CACHE_FILE)):
        raise FileNotFoundError(
            "Cache không tìm thấy! Chạy preprocess_data_asic.py trước."
        )

    print("  Loading cache...")
    X = np.load(X_CACHE_FILE)
    y = np.load(Y_CACHE_FILE)

    # ── Kiểm tra kích thước khớp config ──────────────────────────
    if X.shape[1] != IMG_SIZE or X.shape[2] != IMG_SIZE:
        raise ValueError(
            f"Cache size mismatch: {X.shape[1]}x{X.shape[2]} vs {IMG_SIZE}x{IMG_SIZE}.\n"
            f"Xóa cache và chạy lại preprocess_data_asic.py."
        )

    print(f"\n  Loaded:")
    print(f"    X shape  : {X.shape}")
    print(f"    X dtype  : {X.dtype}")
    print(f"    X range  : [{X.min()}, {X.max()}]")
    print(f"    X memory : {X.nbytes / (1024 * 1024):.2f} MB")
    print(f"    y shape  : {y.shape}")

    # ── Verify ASIC format ────────────────────────────────────────
    print(f"\n  ASIC format check:")
    print(f"    Image size    : {IMG_SIZE}x{IMG_SIZE}  (2^{LOG2_SIZE} x 2^{LOG2_SIZE})")
    print(f"    Data type     : {X.dtype}  (int8)")
    print(f"    Format        : Q0.{FIXED_POINT_FRAC_BITS} fixed-point")
    print(f"    Scale factor  : {SCALE_FACTOR}")
    print(f"    Pixel address : (y << {LOG2_SIZE}) + x")

    # ── Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=SEED,
        test_size=TEST_SIZE,
        stratify=y,
    )

    print(f"\n  Split result:")
    print(f"    Train : {len(X_train)} images  ({100*(1-TEST_SIZE):.0f}%)")
    print(f"    Test  : {len(X_test)} images  ({100*TEST_SIZE:.0f}%)")

    # ── Class distribution ────────────────────────────────────────
    for split_name, y_split in [("Train", y_train), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        print(f"\n  {split_name} distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"    {CATEGORIES[cls]}: {cnt}  ({100 * cnt / len(y_split):.1f}%)")

    # ── Save ──────────────────────────────────────────────────────
    print(f"\n  Saving...")
    np.save(X_TRAIN_FILE, X_train)
    np.save(X_TEST_FILE,  X_test)
    np.save(Y_TRAIN_FILE, y_train)
    np.save(Y_TEST_FILE,  y_test)
    print(f"    {X_TRAIN_FILE}")
    print(f"    {X_TEST_FILE}")
    print(f"    {Y_TRAIN_FILE}")
    print(f"    {Y_TEST_FILE}")

    # ── Memory summary ────────────────────────────────────────────
    train_mb = X_train.nbytes / (1024 * 1024)
    test_mb  = X_test.nbytes  / (1024 * 1024)
    print(f"\n  Memory:")
    print(f"    Train : {train_mb:.2f} MB")
    print(f"    Test  : {test_mb:.2f} MB")
    print(f"    Total : {train_mb + test_mb:.2f} MB")

    # ── ASIC info ─────────────────────────────────────────────────
    print(f"\n  ASIC Hardware Info:")
    print(f"    Frame size    : {X_train[0].nbytes} bytes  ({IMG_SIZE}x{IMG_SIZE})")
    print(f"    DMA block     : {X_train[0].nbytes} bytes")
    print(f"    Pixel address : (y << {LOG2_SIZE}) + x")

    return X_train, X_test, y_train, y_test


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ASIC DRIVER DROWSINESS — TRAIN/TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = split_and_save_asic()

    print("\n  Split completed.")
    print(f"  Data format: Q0.{FIXED_POINT_FRAC_BITS} fixed-point (int8), {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Convert to float khi train: X.astype(float32) / {SCALE_FACTOR}")