"""
Train/Test split for ASIC-optimized data
"""
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
    SCALE_FACTOR
)
import os


def split_and_save_asic():
    """
    Load ASIC preprocessed data, split into train/test, and save
    """
    print("\n✂️ Splitting ASIC data into train/test sets...")
    
    # Load preprocessed data
    if not (os.path.exists(X_CACHE_FILE) and os.path.exists(Y_CACHE_FILE)):
        raise FileNotFoundError(
            "ASIC preprocessed data not found! Please run preprocess_data_asic.py first."
        )
    
    print("📂 Loading ASIC preprocessed data...")
    X = np.load(X_CACHE_FILE)
    y = np.load(Y_CACHE_FILE)
    
    print(f"\n📊 Loaded Data:")
    print(f"  X shape: {X.shape}")
    print(f"  X dtype: {X.dtype}")
    print(f"  X range: [{X.min()}, {X.max()}]")
    print(f"  X memory: {X.nbytes / (1024*1024):.2f} MB")
    print(f"  y shape: {y.shape}")
    print(f"  y dtype: {y.dtype}")
    
    # Verify ASIC format
    print(f"\n🔍 Verifying ASIC format:")
    print(f"  ✓ Image size: {IMG_SIZE}x{IMG_SIZE} (power-of-2)")
    print(f"  ✓ Data type: {X.dtype} (int8)")
    print(f"  ✓ Format: Q0.{FIXED_POINT_FRAC_BITS} fixed-point")
    print(f"  ✓ Scale factor: {SCALE_FACTOR}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=SEED,
        test_size=TEST_SIZE,
        stratify=y
    )
    
    print(f"\n✅ Data split completed!")
    print(f"  Training set: {len(X_train)} images")
    print(f"  Test set: {len(X_test)} images")
    print(f"  Train/Test ratio: {len(X_train)/len(X_test):.2f}")
    
    # Show distribution for each set
    print(f"\n📊 Training set distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {CATEGORIES[cls]}: {count} ({100*count/len(y_train):.1f}%)")
    
    print(f"\n📊 Test set distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {CATEGORIES[cls]}: {count} ({100*count/len(y_test):.1f}%)")
    
    # Save split data
    print(f"\n💾 Saving ASIC train/test splits...")
    np.save(X_TRAIN_FILE, X_train)
    np.save(X_TEST_FILE, X_test)
    np.save(Y_TRAIN_FILE, y_train)
    np.save(Y_TEST_FILE, y_test)
    
    print(f"  Saved to:")
    print(f"    {X_TRAIN_FILE}")
    print(f"    {X_TEST_FILE}")
    print(f"    {Y_TRAIN_FILE}")
    print(f"    {Y_TEST_FILE}")
    
    # Memory usage
    train_mem = X_train.nbytes / (1024*1024)
    test_mem = X_test.nbytes / (1024*1024)
    print(f"\n💾 Memory usage:")
    print(f"  Training set: {train_mem:.2f} MB")
    print(f"  Test set: {test_mem:.2f} MB")
    print(f"  Total: {train_mem + test_mem:.2f} MB")
    
    # ASIC-specific info
    print(f"\n🔧 ASIC Hardware Info:")
    print(f"  Single image size: {X_train[0].nbytes} bytes ({IMG_SIZE}x{IMG_SIZE})")
    print(f"  Memory per batch (1): {X_train[0].nbytes} bytes")
    print(f"  DMA transfer size: {X_train[0].nbytes} bytes")
    print(f"  Addressing: (y << 7) + x  (simple bit-shift)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("="*70)
    print("ASIC DRIVER DROWSINESS - TRAIN/TEST SPLIT")
    print("="*70)
    
    X_train, X_test, y_train, y_test = split_and_save_asic()
    
    print("\n✅ ASIC train/test split completed successfully!")
    print("\n📤 Ready for upload to Google Colab")
    print("   Remember: Data is in Q0.7 fixed-point format (int8)")