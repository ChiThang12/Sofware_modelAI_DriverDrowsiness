"""
Main ASIC-Optimized Preprocessing Pipeline
Combines MediaPipe landmarks with ASIC-friendly data format
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

from config_asic import (
    ACTIVE_DIR,
    FATIGUE_DIR,
    X_CACHE_FILE,
    Y_CACHE_FILE,
    IMG_SIZE,
    CATEGORIES,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    MEDIAPIPE_CONFIG,
    VERBOSE,
    SCALE_FACTOR,
    FIXED_POINT_FRAC_BITS
)
from preprocessing_asic import (
    process_image_asic_friendly,
    verify_asic_friendly,
    rgb_to_grayscale_optimized
)

# MediaPipe setup
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Eye landmarks
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))
all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)


def draw_landmarks(img_dt, face_landmarks, imgW, imgH):
    """Draw face mesh landmarks"""
    image_drawing_tool = img_dt.copy()
    
    connections_drawing_spec = mp_drawing.DrawingSpec(
        thickness=1,
        circle_radius=2,
        color=(255, 255, 255)
    )
    
    mp_drawing.draw_landmarks(
        image=image_drawing_tool,
        landmark_list=face_landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=connections_drawing_spec,
    )
    
    landmarks = face_landmarks.landmark
    for landmark_idx, landmark in enumerate(landmarks):
        if landmark_idx in all_idxs:
            pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
            if pred_cord:
                cv2.circle(image_drawing_tool, pred_cord, 3, (255, 255, 255), -1)
    
    return image_drawing_tool


def process_with_mediapipe_asic(image):
    """
    Process image with MediaPipe landmarks + ASIC optimization
    
    Pipeline:
    1. MediaPipe face mesh (color image required)
    2. Convert to grayscale
    3. Resize to 128x128 (power-of-2)
    4. Normalize to Q0.7 fixed-point [0, 127]
    
    Args:
        image: Input BGR image
    
    Returns:
        int8 image in Q0.7 format, shape (128, 128)
    """
    image = np.ascontiguousarray(image)
    imgH, imgW, _ = image.shape
    
    with mp_facemesh.FaceMesh(**MEDIAPIPE_CONFIG) as face_mesh:
        results = face_mesh.process(image)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks on color image
                processed_img = draw_landmarks(image.copy(), face_landmarks, imgW, imgH)
                
                # Now apply ASIC-friendly processing
                img_asic = process_image_asic_friendly(processed_img)
                return img_asic
    
    # No face detected, process original image
    img_asic = process_image_asic_friendly(image)
    return img_asic


def load_and_preprocess_asic(use_cache=True):
    """
    Load and preprocess all images with ASIC optimization
    
    Returns:
        X: int8 array, shape (N, 128, 128), Q0.7 format [0, 127]
        y: int array, shape (N,), labels
    """
    # Check cache
    if use_cache and os.path.exists(X_CACHE_FILE) and os.path.exists(Y_CACHE_FILE):
        print("📂 ASIC preprocessed data found in cache. Loading...")
        X = np.load(X_CACHE_FILE)
        y = np.load(Y_CACHE_FILE)
        print(f"  Loaded X shape: {X.shape}")
        print(f"  Loaded X dtype: {X.dtype}")
        print(f"  Loaded y shape: {y.shape}")
        
        # Verify first image
        if len(X) > 0:
            verify_asic_friendly(X[0])
        
        return X, y
    
    print("\n" + "="*70)
    print("ASIC-OPTIMIZED PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Target format: Q0.{FIXED_POINT_FRAC_BITS} fixed-point (int8)")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} (power-of-2)")
    print(f"Scale factor: {SCALE_FACTOR} (for conversion to float)")
    print("="*70)
    
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
        
        # Process with progress bar
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
                    processed_img = process_with_mediapipe_asic(roi_color)
                else:
                    # No face detected
                    processed_img = process_image_asic_friendly(img)
                
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
    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=np.int32)
    
    print(f"\n📊 Data Information:")
    print(f"  X shape: {X.shape}")
    print(f"  X dtype: {X.dtype}")
    print(f"  X range: [{X.min()}, {X.max()}]")
    print(f"  X memory: {X.nbytes / (1024*1024):.2f} MB")
    print(f"  y shape: {y.shape}")
    print(f"  y dtype: {y.dtype}")
    
    # Verify ASIC-friendliness
    print(f"\n🔍 Verifying ASIC compatibility...")
    if len(X) > 0:
        verify_asic_friendly(X[0])
    
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
    print("DRIVER DROWSINESS DETECTION - ASIC PREPROCESSING")
    print("="*70)
    
    # Load and preprocess
    X, y = load_and_preprocess_asic(use_cache=True)
    
    print("\n✅ ASIC preprocessing completed successfully!")
    print(f"Final dataset: {len(X)} images")
    print(f"\nData format optimized for ASIC:")
    print(f"  • Fixed-point Q0.{FIXED_POINT_FRAC_BITS} (int8)")
    print(f"  • Power-of-2 dimensions ({IMG_SIZE}x{IMG_SIZE})")
    print(f"  • Bit-shift operations only")
    print(f"  • No floating-point needed")