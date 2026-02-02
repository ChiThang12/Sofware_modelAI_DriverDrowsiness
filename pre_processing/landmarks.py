"""
MediaPipe landmark setup and drawing utilities
"""
import cv2
import numpy as np
import mediapipe as mp
from config import (
    IMG_SIZE, 
    MEDIAPIPE_CONFIG,
    CHOSEN_LEFT_EYE_IDXS,
    CHOSEN_RIGHT_EYE_IDXS
)

# ===================================================================
# MEDIAPIPE SETUP
# ===================================================================
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# ===================================================================
# LANDMARK INDICES
# ===================================================================
# Get all eye landmark indices
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))

all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))

all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
all_chosen_idxs = CHOSEN_LEFT_EYE_IDXS + CHOSEN_RIGHT_EYE_IDXS

print(f"✅ Landmark points configured:")
print(f"  Left eye landmarks: {len(all_left_eye_idxs)}")
print(f"  Right eye landmarks: {len(all_right_eye_idxs)}")
print(f"  Chosen landmarks: {len(all_chosen_idxs)}")


# ===================================================================
# DRAWING FUNCTIONS
# ===================================================================
def draw_landmarks(img_dt, face_landmarks, imgW, imgH):
    """
    Vẽ face mesh landmarks lên ảnh
    
    Args:
        img_dt: Input image
        face_landmarks: MediaPipe face landmarks
        imgW: Image width
        imgH: Image height
    
    Returns:
        Image with drawn landmarks
    """
    image_drawing_tool = img_dt.copy()

    # Drawing specifications
    connections_drawing_spec = mp_drawing.DrawingSpec(
        thickness=1,
        circle_radius=2,
        color=(255, 255, 255)
    )

    # Draw face mesh
    mp_drawing.draw_landmarks(
        image=image_drawing_tool,
        landmark_list=face_landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=connections_drawing_spec,
    )

    # Draw eye landmarks
    landmarks = face_landmarks.landmark
    for landmark_idx, landmark in enumerate(landmarks):
        if landmark_idx in all_idxs:
            pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
            if pred_cord:
                cv2.circle(image_drawing_tool, pred_cord, 3, (255, 255, 255), -1)

    return image_drawing_tool


def process_image_with_landmarks(image):
    """
    Xử lý ảnh với MediaPipe landmarks
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        Processed image with landmarks drawn and resized
    """
    image = np.ascontiguousarray(image)
    imgH, imgW, _ = image.shape

    with mp_facemesh.FaceMesh(**MEDIAPIPE_CONFIG) as face_mesh:
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                processed_img = draw_landmarks(image.copy(), face_landmarks, imgW, imgH)
                resized = cv2.resize(processed_img, (IMG_SIZE, IMG_SIZE))
                return resized

    # Nếu không detect được face, return resized image gốc
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE))