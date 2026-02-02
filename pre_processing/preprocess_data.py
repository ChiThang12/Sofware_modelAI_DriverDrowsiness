"""
Main preprocessing script for Driver Drowsiness Detection
Loads images, applies face detection and MediaPipe landmarks, and saves preprocessed data
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from config import (
    ACTIVE_DIR,
    FATIGUE_DIR,
    X_CACHE_FILE,
    Y_CACHE_FILE,
    IMG_SIZE,
    CATEGORIES,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    VERBOSE,
    PROGRESS_INTERVAL
)
from landmarks import process_image_with_landmarks


def load_and_preprocess_data(use_cache=True):
    """
    Load and preprocess all images from dataset
    
    Args:
        use_cache: If True, load from cache if available
    
    Returns:
        X: Preprocessed images array
        y: Labels array
    """
    # Check cache
    if use_cache and os.path.exists(X_CACHE_FILE) and os.path.exists(Y_CACHE_FILE):
        print("📂 Preprocessed data found in cache. Loading...")
        X = np.load(X_CACHE_FILE)
        y = np.load(Y_CACHE_FILE)
        print(f"  Loaded X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    print("\n📊 Loading & preprocessing data...")
    print("⚠️ This may take 10–15 minutes for ~9,000 images...")

    # Load face detector
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        raise ValueError("Failed to load face cascade classifier!")

    X = []
    y = []
    processed_count = 0
    failed_count = 0

    # Map folders to labels
    category_dirs = {
        0: FATIGUE_DIR,  # Fatigue = 0
        1: ACTIVE_DIR    # Active = 1
    }

    for class_num, category_dir in category_dirs.items():
        category_name = CATEGORIES[class_num]
        print(f"\n📂 Processing {category_name}...")
        
        if not os.path.exists(category_dir):
            print(f"❌ Directory not found: {category_dir}")
            continue

        images = [f for f in os.listdir(category_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_images = len(images)
        print(f"  Found {total_images} images")

        # Use tqdm for progress bar
        for idx, image_name in enumerate(tqdm(images, desc=f"Processing {category_name}")):
            try:
                img_path = os.path.join(category_dir, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if img is None:
                    failed_count += 1
                    continue

                # Detect face
                faces = face_cascade.detectMultiScale(
                    img, 
                    FACE_DETECTION_SCALE_FACTOR, 
                    FACE_DETECTION_MIN_NEIGHBORS
                )

                # Process image
                if len(faces) > 0:
                    (x, y_coord, w, h) = faces[0]
                    roi_color = img[y_coord:y_coord+h, x:x+w]
                    processed_img = process_image_with_landmarks(roi_color)
                else:
                    # No face detected, just resize
                    processed_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                X.append(processed_img)
                y.append(class_num)
                processed_count += 1

            except Exception as e:
                if VERBOSE and failed_count < 10:
                    print(f"  Error processing {image_name}: {str(e)}")
                failed_count += 1
                continue

    print(f"\n✅ Preprocessing complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Failed: {failed_count} images")

    # Convert to numpy arrays
    X = np.array(X, dtype="float32")
    y = np.array(y)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Save to cache
    print(f"\n💾 Saving to cache...")
    np.save(X_CACHE_FILE, X)
    np.save(Y_CACHE_FILE, y)
    print(f"  Saved to:")
    print(f"    {X_CACHE_FILE}")
    print(f"    {Y_CACHE_FILE}")

    # Show distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {CATEGORIES[cls]}: {count} images ({100 * count / len(y):.1f}%)")

    return X, y


if __name__ == "__main__":
    print("="*70)
    print("DRIVER DROWSINESS DETECTION - DATA PREPROCESSING")
    print("="*70)
    
    # Load and preprocess
    X, y = load_and_preprocess_data(use_cache=True)
    
    print("\n✅ Preprocessing completed successfully!")
    print(f"Final dataset: {len(X)} images")