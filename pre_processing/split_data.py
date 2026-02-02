"""
Train/Test data splitting script
"""
import numpy as np
import os    
from sklearn.model_selection import train_test_split
from config import (
    X_CACHE_FILE,
    Y_CACHE_FILE,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    SEED,
    TEST_SIZE,
    CATEGORIES
)


def split_and_save_data():
    """
    Load preprocessed data, split into train/test, and save
    """
    print("\n✂️ Splitting data into train/test sets...")
    
    # Load preprocessed data
    if not (os.path.exists(X_CACHE_FILE) and os.path.exists(Y_CACHE_FILE)):
        raise FileNotFoundError(
            "Preprocessed data not found! Please run preprocess_data.py first."
        )
    
    print("📂 Loading preprocessed data...")
    X = np.load(X_CACHE_FILE)
    y = np.load(Y_CACHE_FILE)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
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
    print(f"\n💾 Saving train/test splits...")
    np.save(X_TRAIN_FILE, X_train)
    np.save(X_TEST_FILE, X_test)
    np.save(Y_TRAIN_FILE, y_train)
    np.save(Y_TEST_FILE, y_test)
    
    print(f"  Saved to:")
    print(f"    {X_TRAIN_FILE}")
    print(f"    {X_TEST_FILE}")
    print(f"    {Y_TRAIN_FILE}")
    print(f"    {Y_TEST_FILE}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import os
    
    print("="*70)
    print("DRIVER DROWSINESS DETECTION - TRAIN/TEST SPLIT")
    print("="*70)
    
    X_train, X_test, y_train, y_test = split_and_save_data()
    
    print("\n✅ Train/Test split completed successfully!")