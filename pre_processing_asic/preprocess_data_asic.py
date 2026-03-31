"""
Main ASIC-Optimized Preprocessing Pipeline — Eye Crop Version
Thay vì dùng toàn bộ khuôn mặt, pipeline này:
  1. Dùng Haar cascade detect mặt → crop ROI
  2. Dùng MediaPipe FaceMesh detect landmark mắt trái + phải
  3. Crop từng mắt riêng lẻ (có padding), convert grayscale
  4. Resize mỗi mắt về 64×32, stack dọc → ảnh vuông 64×64
  5. Normalize uint8 >> 1 → int8 Q0.7 [0, 127]

Output: int8 array shape (N, 64, 64), Q0.7 fixed-point [0, 127]

Layout ảnh đầu ra:
  ┌──────────────┐
  │  Mắt trái   │  (32 hàng trên)
  ├──────────────┤
  │  Mắt phải   │  (32 hàng dưới)
  └──────────────┘
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
    FIXED_POINT_FRAC_BITS,
    QUANTIZATION_MAX,
)
from preprocessing_asic import verify_asic_friendly

# ===================================================================
# CONSTANTS
# ===================================================================
# Mỗi mắt được resize về 64×32, stack dọc thành 64×64
EYE_W       = IMG_SIZE        # 64 — chiều rộng mỗi mắt
EYE_H       = IMG_SIZE // 2   # 32 — chiều cao mỗi mắt
# Padding tính theo tỉ lệ bounding box của mắt
# 0.30 = thêm 30% mỗi phía → đảm bảo không bị crop sát mi mắt
EYE_PADDING = 0.30

# ===================================================================
# MEDIAPIPE SETUP
# ===================================================================
mp_facemesh = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Index các landmark thuộc mắt trái và mắt phải
# (theo hệ tọa độ MediaPipe — trái/phải theo góc nhìn của người)
all_left_eye_idxs  = set(np.ravel(list(mp_facemesh.FACEMESH_LEFT_EYE)))
all_right_eye_idxs = set(np.ravel(list(mp_facemesh.FACEMESH_RIGHT_EYE)))


# ===================================================================
# CORE: CROP TỪNG MẮT VÀ STACK
# ===================================================================
def _get_eye_bbox(landmarks, eye_idxs, imgW, imgH, padding=EYE_PADDING):
    """
    Tính bounding box của một mắt từ tập landmark indices.

    Args:
        landmarks : face_landmarks.landmark (list MediaPipe)
        eye_idxs  : set of landmark indices thuộc mắt đó
        imgW, imgH: kích thước ảnh gốc (pixel)
        padding   : tỉ lệ padding thêm vào mỗi phía

    Returns:
        (x1, y1, x2, y2) trong tọa độ pixel, đã clamp trong ảnh
        Trả về None nếu không có landmark hợp lệ
    """
    xs, ys = [], []
    for idx, lm in enumerate(landmarks):
        if idx not in eye_idxs:
            continue
        coord = denormalize_coordinates(lm.x, lm.y, imgW, imgH)
        if coord is None:
            continue
        xs.append(coord[0])
        ys.append(coord[1])

    if not xs:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Thêm padding theo tỉ lệ kích thước bounding box
    pw = int((x_max - x_min) * padding)
    ph = int((y_max - y_min) * padding)

    x1 = max(0,    x_min - pw)
    y1 = max(0,    y_min - ph)
    x2 = min(imgW, x_max + pw)
    y2 = min(imgH, y_max + ph)

    # Bounding box quá nhỏ → không tin cậy
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def _crop_and_prepare_eye(img_gray, bbox):
    """
    Crop vùng mắt từ ảnh grayscale rồi resize về EYE_W × EYE_H.

    Args:
        img_gray: grayscale image (H, W), dtype uint8
        bbox    : (x1, y1, x2, y2) tọa độ pixel

    Returns:
        uint8 array (EYE_H, EYE_W) — ảnh mắt đã resize
    """
    x1, y1, x2, y2 = bbox
    crop = img_gray[y1:y2, x1:x2]
    # INTER_AREA tốt nhất khi downscale — giảm aliasing
    resized = cv2.resize(crop, (EYE_W, EYE_H), interpolation=cv2.INTER_AREA)
    return resized


def crop_both_eyes_stacked(image, face_landmarks):
    """
    Crop mắt trái + mắt phải, stack dọc thành ảnh vuông 64×64.

    Pipeline:
        1. Convert BGR → Grayscale
        2. Lấy bbox mắt trái từ landmark → crop → resize 64×32
        3. Lấy bbox mắt phải từ landmark → crop → resize 64×32
        4. Stack dọc: [mắt trái / mắt phải] → (64, 64) uint8
        5. uint8 >> 1 → int8 Q0.7 [0, 127]

    Args:
        image         : BGR image (H, W, 3) — ROI khuôn mặt
        face_landmarks: MediaPipe face landmark object

    Returns:
        int8 array (64, 64) — Q0.7 fixed-point
        None nếu không lấy được cả hai mắt
    """
    imgH, imgW = image.shape[:2]

    # Bước 1: Grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lms = face_landmarks.landmark

    # Bước 2 & 3: Lấy bbox từng mắt
    bbox_left  = _get_eye_bbox(lms, all_left_eye_idxs,  imgW, imgH)
    bbox_right = _get_eye_bbox(lms, all_right_eye_idxs, imgW, imgH)

    if bbox_left is None or bbox_right is None:
        return None

    eye_left  = _crop_and_prepare_eye(img_gray, bbox_left)   # (32, 64) uint8
    eye_right = _crop_and_prepare_eye(img_gray, bbox_right)  # (32, 64) uint8

    # Bước 4: Stack dọc → (64, 64) uint8
    stacked = np.vstack([eye_left, eye_right])

    # Bước 5: Normalize → int8 Q0.7 [0, 127]
    fixed = (stacked.astype(np.int16) >> 1).astype(np.int8)
    fixed = np.clip(fixed, 0, QUANTIZATION_MAX)

    return fixed


# ===================================================================
# MEDIAPIPE WRAPPER
# ===================================================================
def process_with_eye_crop(image):
    """
    Chạy MediaPipe FaceMesh trên ROI khuôn mặt, crop hai mắt,
    trả về ảnh vuông 64×64 int8 Q0.7.

    Args:
        image: BGR image (H, W, 3) — ROI khuôn mặt đã crop bởi Haar

    Returns:
        int8 array (64, 64) — Q0.7 fixed-point
        None nếu không detect được mesh hoặc không crop được mắt
    """
    image = np.ascontiguousarray(image)

    with mp_facemesh.FaceMesh(**MEDIAPIPE_CONFIG) as face_mesh:
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                result = crop_both_eyes_stacked(image, face_landmarks)
                if result is not None:
                    return result

    # Không detect được mesh hoặc không crop được mắt → bỏ ảnh
    return None


# ===================================================================
# MAIN PREPROCESSING FUNCTION
# ===================================================================
def load_and_preprocess_asic(use_cache=True):
    """
    Load và preprocess toàn bộ dataset — eye crop version.

    Returns:
        X : int8 array, shape (N, 64, 64), Q0.7 [0, 127]
            Layout: hàng 0–31 = mắt trái, hàng 32–63 = mắt phải
        y : int32 array, shape (N,)  — 0=Fatigue, 1=Active
    """
    # ── Load cache nếu có ─────────────────────────────────────────
    if use_cache and os.path.exists(X_CACHE_FILE) and os.path.exists(Y_CACHE_FILE):
        print("  Cached data found. Loading...")
        X = np.load(X_CACHE_FILE)
        y = np.load(Y_CACHE_FILE)

        print(f"  X shape : {X.shape}")
        print(f"  X dtype : {X.dtype}")
        print(f"  y shape : {y.shape}")

        if X.shape[1] != IMG_SIZE or X.shape[2] != IMG_SIZE:
            print(f"\n  Cache size mismatch: {X.shape[1]}x{X.shape[2]} vs {IMG_SIZE}x{IMG_SIZE}")
            print(f"  Xóa cache cũ và preprocess lại...")
            os.remove(X_CACHE_FILE)
            os.remove(Y_CACHE_FILE)
            return load_and_preprocess_asic(use_cache=False)

        if len(X) > 0:
            verify_asic_friendly(X[0])
        return X, y

    # ── Preprocessing từ đầu ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("ASIC PREPROCESSING PIPELINE — EYE CROP MODE")
    print("=" * 70)
    print(f"Output size   : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  └─ Mắt trái : {EYE_W}x{EYE_H}  (hàng   0–{EYE_H-1})")
    print(f"  └─ Mắt phải : {EYE_W}x{EYE_H}  (hàng {EYE_H}–{IMG_SIZE-1})")
    print(f"Eye padding   : {EYE_PADDING*100:.0f}% mỗi phía")
    print(f"Output format : Q0.{FIXED_POINT_FRAC_BITS} fixed-point (int8)")
    print(f"Scale factor  : {SCALE_FACTOR}")
    print("=" * 70)

    # Load Haar cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise ValueError("Không load được Haar cascade classifier!")

    X = []
    y = []
    processed_count = 0
    failed_count    = 0
    no_face_count   = 0
    no_eye_count    = 0

    category_dirs = {
        0: FATIGUE_DIR,
        1: ACTIVE_DIR,
    }

    for class_num, category_dir in category_dirs.items():
        category_name = CATEGORIES[class_num]
        print(f"\n  Processing {category_name}...")

        if not os.path.exists(category_dir):
            print(f"  Directory not found: {category_dir}")
            continue

        images = [f for f in os.listdir(category_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"  Found {len(images)} images")

        for image_name in tqdm(images, desc=f"  {category_name}"):
            try:
                img_path = os.path.join(category_dir, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if img is None:
                    failed_count += 1
                    continue

                # ── Bước 1: Haar cascade detect mặt ──────────────
                faces = face_cascade.detectMultiScale(
                    img,
                    FACE_DETECTION_SCALE_FACTOR,
                    FACE_DETECTION_MIN_NEIGHBORS,
                )

                if len(faces) > 0:
                    x_f, y_f, w_f, h_f = faces[0]
                    roi = img[y_f:y_f + h_f, x_f:x_f + w_f]
                else:
                    # Không detect mặt → thử MediaPipe trên toàn ảnh
                    no_face_count += 1
                    roi = img

                # ── Bước 2–5: MediaPipe + crop mắt + quantize ────
                processed_img = process_with_eye_crop(roi)

                if processed_img is None:
                    # Không lấy được mắt → bỏ ảnh
                    no_eye_count += 1
                    continue

                X.append(processed_img)
                y.append(class_num)
                processed_count += 1

            except Exception as e:
                if VERBOSE and failed_count < 10:
                    print(f"  Error [{image_name}]: {e}")
                failed_count += 1

    print(f"\n  Processed  : {processed_count}")
    print(f"  No face    : {no_face_count}  (thử toàn ảnh)")
    print(f"  No eye     : {no_eye_count}   (bỏ — không crop được mắt)")
    print(f"  Failed     : {failed_count}")

    if processed_count == 0:
        raise ValueError("Không xử lý được ảnh nào! Kiểm tra lại dataset và thư mục.")

    # ── Convert to numpy ──────────────────────────────────────────
    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=np.int32)

    print(f"\n  X shape  : {X.shape}")
    print(f"  X dtype  : {X.dtype}")
    print(f"  X range  : [{X.min()}, {X.max()}]")
    print(f"  X memory : {X.nbytes / (1024 * 1024):.2f} MB")

    # ── Verify ASIC format ────────────────────────────────────────
    print(f"\n  Verifying ASIC format...")
    if len(X) > 0:
        verify_asic_friendly(X[0])

    # ── Class distribution ────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {CATEGORIES[cls]}: {cnt} ({100 * cnt / len(y):.1f}%)")

    # ── Save cache ────────────────────────────────────────────────
    print(f"\n  Saving cache...")
    np.save(X_CACHE_FILE, X)
    np.save(Y_CACHE_FILE, y)
    print(f"    {X_CACHE_FILE}")
    print(f"    {Y_CACHE_FILE}")

    return X, y


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DRIVER DROWSINESS DETECTION — ASIC EYE CROP PREPROCESSING")
    print("=" * 70)

    X, y = load_and_preprocess_asic(use_cache=True)

    print(f"\n  Eye crop preprocessing completed: {len(X)} images")
    print(f"  Layout : hàng 0–{EYE_H-1} = mắt trái  |  hàng {EYE_H}–{IMG_SIZE-1} = mắt phải")
    print(f"  Format : Q0.{FIXED_POINT_FRAC_BITS} fixed-point (int8), {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Convert to float khi train: X.astype(float32) / {SCALE_FACTOR}")