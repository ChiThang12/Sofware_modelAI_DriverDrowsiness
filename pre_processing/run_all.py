"""
Master script to run all preprocessing steps
"""
import os
import sys
import time
from datetime import datetime


def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'cv2',
        'numpy',
        'mediapipe',
        'sklearn',
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install them with: pip install {' '.join(missing)}")
        print(f"Note: for cv2, install with: pip install opencv-python")
        return False
    
    print("✅ All dependencies installed!\n")
    return True


def check_dataset():
    """Check if dataset directories exist"""
    from config import ACTIVE_DIR, FATIGUE_DIR
    
    print("🔍 Checking dataset...")
    
    if not os.path.exists(ACTIVE_DIR):
        print(f"❌ Active subjects directory not found: {ACTIVE_DIR}")
        return False
    
    if not os.path.exists(FATIGUE_DIR):
        print(f"❌ Fatigue subjects directory not found: {FATIGUE_DIR}")
        return False
    
    active_count = len([f for f in os.listdir(ACTIVE_DIR) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    fatigue_count = len([f for f in os.listdir(FATIGUE_DIR) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"  ✅ Active subjects: {active_count} images")
    print(f"  ✅ Fatigue subjects: {fatigue_count} images")
    print(f"  ✅ Total: {active_count + fatigue_count} images\n")
    
    return True


def run_preprocessing():
    """Run the preprocessing pipeline"""
    print("="*70)
    print("DRIVER DROWSINESS DETECTION - FULL PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return False
    
    # Step 2: Check dataset
    if not check_dataset():
        return False
    
    # Step 3: Preprocess data
    print("="*70)
    print("STEP 1: PREPROCESSING DATA")
    print("="*70)
    from preprocess_data import load_and_preprocess_data
    
    try:
        X, y = load_and_preprocess_data(use_cache=True)
        print(f"\n✅ Step 1 completed: {len(X)} images preprocessed")
    except Exception as e:
        print(f"\n❌ Error in preprocessing: {str(e)}")
        return False
    
    # Step 4: Split data
    print("\n" + "="*70)
    print("STEP 2: SPLITTING TRAIN/TEST DATA")
    print("="*70)
    from split_data import split_and_save_data
    
    try:
        X_train, X_test, y_train, y_test = split_and_save_data()
        print(f"\n✅ Step 2 completed")
    except Exception as e:
        print(f"\n❌ Error in splitting: {str(e)}")
        return False
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE COMPLETED!")
    print("="*70)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    from config import OUTPUT_DIR
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
    print(f"\nFiles ready to upload to Google Colab:")
    print(f"  • X_train.npy ({X_train.shape})")
    print(f"  • X_test.npy ({X_test.shape})")
    print(f"  • y_train.npy ({y_train.shape})")
    print(f"  • y_test.npy ({y_test.shape})")
    
    return True


if __name__ == "__main__":
    success = run_preprocessing()
    
    if success:
        print("\n✅ All preprocessing completed successfully!")
        print("\n📤 Next steps:")
        print("  1. Upload the .npy files from PreprocessedData folder to Google Drive")
        print("  2. Load them in Google Colab for training")
        sys.exit(0)
    else:
        print("\n❌ Preprocessing failed. Please check errors above.")
        sys.exit(1)