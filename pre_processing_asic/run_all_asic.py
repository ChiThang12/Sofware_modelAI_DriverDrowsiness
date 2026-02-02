"""
Master script for ASIC-optimized preprocessing
Runs full pipeline: preprocessing -> split -> verify
"""
import os
import sys
import time
from datetime import datetime
from config_asic import OUTPUT_DIR, IMG_SIZE, FIXED_POINT_FRAC_BITS

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['cv2', 'numpy', 'mediapipe', 'sklearn', 'tqdm']
    
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
        print(f"Install: pip install {' '.join(missing)}")
        print(f"Note: for cv2, use: pip install opencv-python")
        return False
    
    print("✅ All dependencies installed!\n")
    return True


def check_dataset():
    """Check if dataset directories exist"""
    from config_asic import ACTIVE_DIR, FATIGUE_DIR
    
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


def display_asic_config():
    """Display ASIC configuration"""
    from config_asic import (
        IMG_SIZE, FIXED_POINT_FRAC_BITS, SCALE_FACTOR,
        QUANTIZATION_BITS, USE_FIXED_POINT
    )
    
    print("="*70)
    print("ASIC CONFIGURATION")
    print("="*70)
    print(f"Image Size:          {IMG_SIZE}x{IMG_SIZE} (power-of-2)")
    print(f"Fixed-Point Format:  Q0.{FIXED_POINT_FRAC_BITS}")
    print(f"Quantization Bits:   {QUANTIZATION_BITS}-bit")
    print(f"Scale Factor:        {SCALE_FACTOR} (2^{FIXED_POINT_FRAC_BITS})")
    print(f"Use Fixed-Point:     {USE_FIXED_POINT}")
    print(f"\nHardware Benefits:")
    print(f"  • Bit-shift normalization (x >> {FIXED_POINT_FRAC_BITS})")
    print(f"  • Power-of-2 addressing")
    print(f"  • No FPU required")
    print(f"  • Minimal memory: {IMG_SIZE * IMG_SIZE / 1024:.1f} KB/image")
    print("="*70 + "\n")


def run_asic_preprocessing():
    """Run the full ASIC preprocessing pipeline"""
    print("="*70)
    print("ASIC DRIVER DROWSINESS - FULL PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return False
    
    # Step 2: Check dataset
    if not check_dataset():
        return False
    
    # Step 3: Display ASIC config
    display_asic_config()
    
    # Step 4: Preprocess data
    print("="*70)
    print("STEP 1: ASIC-OPTIMIZED PREPROCESSING")
    print("="*70)
    from preprocess_data_asic import load_and_preprocess_asic
    
    try:
        X, y = load_and_preprocess_asic(use_cache=True)
        print(f"\n✅ Step 1 completed: {len(X)} images preprocessed")
        print(f"   Format: Q0.7 fixed-point (int8)")
        print(f"   Size: {IMG_SIZE}x{IMG_SIZE} per image")
    except Exception as e:
        print(f"\n❌ Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Split data
    print("\n" + "="*70)
    print("STEP 2: SPLITTING TRAIN/TEST DATA")
    print("="*70)
    from split_data_asic import split_and_save_asic
    
    try:
        X_train, X_test, y_train, y_test = split_and_save_asic()
        print(f"\n✅ Step 2 completed")
    except Exception as e:
        print(f"\n❌ Error in splitting: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Verify ASIC format
    print("\n" + "="*70)
    print("STEP 3: VERIFYING ASIC FORMAT")
    print("="*70)
    from preprocessing_asic import verify_asic_friendly, simulate_asic_inference
    
    try:
        print("\nVerifying training data sample...")
        verify_asic_friendly(X_train[0])
        
        print("\nSimulating ASIC hardware...")
        simulate_asic_inference(X_train[0])
        
        print(f"\n✅ Step 3 completed")
    except Exception as e:
        print(f"\n❌ Error in verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("ASIC PREPROCESSING PIPELINE COMPLETED!")
    print("="*70)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
    print(f"\n📦 Files ready for Google Colab:")
    print(f"  • X_train_asic.npy ({X_train.shape})")
    print(f"  • X_test_asic.npy ({X_test.shape})")
    print(f"  • y_train_asic.npy ({y_train.shape})")
    print(f"  • y_test_asic.npy ({y_test.shape})")
    
    print(f"\n🎯 ASIC-Optimized Format:")
    print(f"  • Q0.{FIXED_POINT_FRAC_BITS} fixed-point (int8)")
    print(f"  • {IMG_SIZE}x{IMG_SIZE} images (power-of-2)")
    print(f"  • Range: [0, 127]")
    print(f"  • Memory: {X_train[0].nbytes} bytes/image")
    
    print(f"\n💡 Hardware Benefits:")
    print(f"  ✓ Bit-shift operations only")
    print(f"  ✓ No multiply for normalization")
    print(f"  ✓ Power-of-2 indexing")
    print(f"  ✓ Fixed-point arithmetic")
    print(f"  ✓ Minimal memory footprint")
    
    print(f"\n📈 Performance Estimates:")
    cycles_per_image = IMG_SIZE * IMG_SIZE  # Simplified
    clock_freq = 200e6  # 200 MHz
    latency_us = (cycles_per_image / clock_freq) * 1e6
    fps = 1e6 / latency_us
    print(f"  @ 200 MHz:")
    print(f"    Latency: ~{latency_us:.1f} µs/image")
    print(f"    Throughput: ~{fps:.0f} FPS")
    
    return True


if __name__ == "__main__":
    # Import here to show config early
    from config_asic import IMG_SIZE
    
    success = run_asic_preprocessing()
    
    if success:
        print("\n✅ All ASIC preprocessing completed successfully!")
        print("\n📤 Next steps:")
        print("  1. Upload .npy files from PreprocessedData_ASIC to Google Drive")
        print("  2. Use colab_loader_asic.py to load data in Colab")
        print("  3. Train model with quantization-aware training")
        print("  4. Convert to TFLite INT8")
        print("  5. Implement in RTL for ASIC")
        sys.exit(0)
    else:
        print("\n❌ ASIC preprocessing failed. Please check errors above.")
        sys.exit(1)