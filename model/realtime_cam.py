"""
MODULE 1 — Realtime Webcam Detection
=====================================
Dùng laptop camera để detect drowsiness theo thời gian thực.

Pipeline (GIỐNG HỆT lúc train):
  Camera frame (BGR)
  → Detect face (Haar Cascade)
  → Crop ROI
  → MediaPipe FaceMesh (vẽ landmarks)
  → Grayscale
  → Resize 128×128
  → >> 1  (uint8 → Q0.7 int8, range [0,127])
  → / 128.0 (int8 → float32 cho TFLite input)
  → INT8 TFLite inference
  → Hiển thị kết quả

Logic cảnh báo BUỒN NGỦ (2 điều kiện cùng phải thỏa):
  1. Model trả về FATIGUE với confidence = 100%  (prob_active = 0.0)
  2. Trạng thái FATIGUE liên tục kéo dài ≥ DROWSY_DURATION giây (mặc định 0.5s)
     → nhằm phân biệt với nháy mắt bình thường

Cài đặt:
  pip install opencv-python mediapipe numpy tflite-runtime
  (hoặc pip install tensorflow nếu đã có)

Chạy:
  python realtime_cam.py --model drowsiness_asic_int8.tflite
  python realtime_cam.py --model drowsiness_asic_int8.tflite --duration 0.8
"""

import cv2
import numpy as np
import argparse
import time
import sys

# ── Load TFLite runtime ─────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("✅ Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("✅ Using tensorflow.lite")

# ── MediaPipe ───────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    mp_facemesh  = mp.solutions.face_mesh
    mp_drawing   = mp.solutions.drawing_utils
    denorm_coord = mp_drawing._normalized_to_pixel_coordinates
    MEDIAPIPE_OK = True

    _left    = set(np.ravel(list(mp_facemesh.FACEMESH_LEFT_EYE)))
    _right   = set(np.ravel(list(mp_facemesh.FACEMESH_RIGHT_EYE)))
    EYE_IDXS = _left | _right
except ImportError:
    MEDIAPIPE_OK = False
    print("⚠️  MediaPipe không có — chỉ dùng Haar Cascade")


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
IMG_SIZE         = 128
SCALE_FACTOR     = 128.0    # 2^7

# Dựa trên calibration thực tế:
#   Mắt MỞ  (ACTIVE)  → prob ≈ 0.05 ~ 0.95  (logit ~ -4 đến +2)
#   Mắt NHẮM (FATIGUE) → prob ≈ 0.0002~0.003 (logit ~ -6 đến -8)
#
# → prob THẤP = mắt NHẮM → FATIGUE
# → Ngưỡng: prob < FATIGUE_THRESHOLD thì coi là FATIGUE
FATIGUE_THRESHOLD = 0.01    # prob < 0.01 → mắt NHẮM → FATIGUE

# Thời gian nhắm mắt tối thiểu để kích hoạt cảnh báo (giây)
# Dưới ngưỡng này coi là nháy mắt bình thường
DEFAULT_DROWSY_DURATION = 0.5

TARGET_FPS       = 30
INFERENCE_STRIDE = 3        # Inference mỗi N frames để giảm CPU


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (bản sao chính xác từ preprocessing_asic.py)
# ════════════════════════════════════════════════════════════════════════════

def draw_landmarks_on_roi(roi_bgr):
    """Vẽ FaceMesh landmarks lên ROI màu (giống draw_landmarks() lúc train)"""
    if not MEDIAPIPE_OK:
        return roi_bgr

    h, w = roi_bgr.shape[:2]
    out  = roi_bgr.copy()
    conn_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2,
                                       color=(255, 255, 255))
    with mp_facemesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=False,
                               min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=out,
                    landmark_list=fl,
                    connections=mp_facemesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=conn_spec,
                )
                for idx, lm in enumerate(fl.landmark):
                    if idx in EYE_IDXS:
                        pt = denorm_coord(lm.x, lm.y, w, h)
                        if pt:
                            cv2.circle(out, pt, 3, (255, 255, 255), -1)
    return out


def preprocess_frame(roi_bgr):
    """
    Chuyển ROI → int8 Q0.7 → float32 sẵn cho TFLite.

    Returns:
        fp32_input : float32 (1,128,128,1) — đưa vào TFLite
        q07        : int8   (128,128)      — để debug/preview
    """
    with_lm = draw_landmarks_on_roi(roi_bgr)
    gray    = cv2.cvtColor(with_lm, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    q07     = np.clip((resized.astype(np.int16) >> 1), 0, 127).astype(np.int8)
    fp32    = q07.astype(np.float32) / SCALE_FACTOR
    inp     = fp32[np.newaxis, :, :, np.newaxis]   # (1,128,128,1)
    return inp, q07


# ════════════════════════════════════════════════════════════════════════════
# TFLITE INFERENCE
# ════════════════════════════════════════════════════════════════════════════

class DrowsinessDetector:
    def __init__(self, model_path):
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()

        self.inp_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]

        self.inp_scale = float(self.inp_det['quantization_parameters']['scales'][0])
        self.inp_zp    = int(self.inp_det['quantization_parameters']['zero_points'][0])
        self.out_scale = float(self.out_det['quantization_parameters']['scales'][0])
        self.out_zp    = int(self.out_det['quantization_parameters']['zero_points'][0])

        print(f"\n🔢 TFLite quantization params:")
        print(f"   Input  scale={self.inp_scale:.6f}, zp={self.inp_zp}")
        print(f"   Output scale={self.out_scale:.6f}, zp={self.out_zp}")
        print(f"   Input  dtype : {self.inp_det['dtype']}")
        print(f"   Output dtype : {self.out_det['dtype']}")

    def predict(self, fp32_input):
        """
        Args:
            fp32_input: float32 (1,128,128,1) trong khoảng [0,1]

        Returns:
            prob_active: float [0,1] — xác suất ACTIVE (mắt mở)
            logit      : float       — raw logit trước sigmoid
        """
        inp_int8 = np.round(fp32_input / self.inp_scale + self.inp_zp
                            ).clip(-128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp_det['index'], inp_int8)
        self.interp.invoke()
        out   = self.interp.get_tensor(self.out_det['index'])
        logit = float((int(out[0][0]) - self.out_zp) * self.out_scale)
        # prob thấp (< 0.01) = logit rất âm = mắt NHẮM = FATIGUE
        # prob cao            = logit dương/âm nhỏ = mắt MỞ = ACTIVE
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob, logit


# ════════════════════════════════════════════════════════════════════════════
# DROWSINESS STATE MACHINE
# ════════════════════════════════════════════════════════════════════════════

class DrowsinessGuard:
    """
    Lọc cảnh báo buồn ngủ theo 2 điều kiện:
      1. prob < FATIGUE_THRESHOLD (0.01) → mắt nhắm
      2. Trạng thái đó phải kéo dài liên tục ≥ min_duration giây
         (phân biệt với nháy mắt bình thường)

    Trạng thái:
      NORMAL   — mắt mở  (prob >= 0.01)
      WATCHING — mắt nhắm nhưng chưa đủ thời gian
      ALERT    — đã nhắm đủ thời gian → cảnh báo buồn ngủ
    """

    def __init__(self, min_duration=DEFAULT_DROWSY_DURATION):
        self.min_duration   = min_duration
        self.fatigue_since  = None   # Thời điểm bắt đầu chuỗi FATIGUE 100%
        self.alert_active   = False  # Đang trong trạng thái cảnh báo

    def update(self, prob_active):
        """
        Cập nhật state machine với kết quả inference mới.

        Args:
            prob_active: float [0,1]

        Returns:
            state   : 'NORMAL' | 'WATCHING' | 'ALERT'
            elapsed : float — số giây đã nhắm mắt liên tục (0 nếu NORMAL)
        """
        now = time.time()

        # prob < 0.01 → logit rất âm → mắt NHẮM → FATIGUE
        is_fatigue = (prob_active < FATIGUE_THRESHOLD)

        if is_fatigue:
            if self.fatigue_since is None:
                self.fatigue_since = now    # Bắt đầu chuỗi mới

            elapsed = now - self.fatigue_since

            if elapsed >= self.min_duration:
                self.alert_active = True
                return 'ALERT', elapsed
            else:
                return 'WATCHING', elapsed
        else:
            # Reset — bất kỳ frame nào không đạt 100% → phá chuỗi
            self.fatigue_since = None
            self.alert_active  = False
            return 'NORMAL', 0.0


# ════════════════════════════════════════════════════════════════════════════
# MAIN REALTIME LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_realtime(model_path, cam_id=0, show_preprocessing=True,
                 drowsy_duration=DEFAULT_DROWSY_DURATION):
    detector = DrowsinessDetector(model_path)
    guard    = DrowsinessGuard(min_duration=drowsy_duration)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"❌ Không mở được camera {cam_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    frame_time = 1.0 / TARGET_FPS

    print(f"\n🎥 Camera đang chạy — nhấn Q để thoát, S để chụp ảnh\n")
    print(f"⚡ FPS target      : {TARGET_FPS}")
    print(f"🎯 Cảnh báo khi   : FATIGUE = 100% liên tục ≥ {drowsy_duration}s\n")

    fps_buf = []
    prob    = 0.5     # khởi tạo trung tính
    state   = 'NORMAL'
    elapsed = 0.0
    frame_i = 0
    q07     = None

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("❌ Không đọc được frame — camera bị ngắt?")
                break

            display = frame.copy()
            gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_fr, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = frame[y:y+h, x:x+w]

                # Inference mỗi INFERENCE_STRIDE frames (giảm CPU)
                # nhưng guard.update() gọi mỗi frame để state luôn chính xác
                if frame_i % INFERENCE_STRIDE == 0:
                    fp32_inp, q07 = preprocess_frame(roi)
                    prob, logit   = detector.predict(fp32_inp)
                    # DEBUG — xóa sau khi verify
                    print(f"[frame {frame_i:5d}] logit={logit:+.4f}  prob={prob:.4f}  state={state}")

                state, elapsed = guard.update(prob)

                # ── Quyết định hiển thị theo state ─────────────────────

                if state == 'ALERT':
                    # ⚠️ CẢNH BÁO BUỒN NGỦ — overlay đỏ toàn màn hình
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0),
                                  (display.shape[1], display.shape[0]),
                                  (0, 0, 180), -1)
                    cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(display, f"⚠ BUON NGU! ({elapsed:.1f}s)",
                                (x, y - 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)
                    cv2.putText(display,
                                "CANH BAO: BUON NGU! Hay dung xe!",
                                (display.shape[1]//2 - 250, display.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

                elif state == 'WATCHING':
                    progress = min(elapsed / drowsy_duration, 1.0)
                    bar_w    = int(w * progress)
                    fat_pct  = (1.0 - prob) * 100   # prob thấp → fatigue cao

                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 140, 255), 2)
                    cv2.putText(display,
                                f"FATIGUE {fat_pct:.1f}% ({elapsed:.2f}s / {drowsy_duration}s)",
                                (x, y - 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 140, 255), 2)
                    cv2.rectangle(display, (x, y+h+4), (x + bar_w, y+h+12),
                                  (0, 140, 255), -1)
                    cv2.rectangle(display, (x, y+h+4), (x+w, y+h+12),
                                  (80, 80, 80), 1)

                else:  # NORMAL — mắt mở (prob >= 0.01)
                    active_pct = prob * 100   # prob cao = mắt mở rõ
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 0), 2)
                    cv2.putText(display,
                                f"ACTIVE ({active_pct:.1f}%)",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 220, 0), 2)

                # Q0.7 preview — góc trên phải
                if show_preprocessing and q07 is not None:
                    prev_u8  = np.clip(q07.astype(np.int16) * 2, 0, 255).astype(np.uint8)
                    prev_bgr = cv2.cvtColor(cv2.resize(prev_u8, (128, 128)),
                                            cv2.COLOR_GRAY2BGR)
                    x1 = display.shape[1] - 138
                    display[10:138, x1:x1 + 128] = prev_bgr
                    cv2.putText(display, "Q0.7 preview", (x1, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            else:
                cv2.putText(display, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                # Mất mặt → reset guard + prob về 0 (mắt mở)
                guard.fatigue_since = None
                guard.alert_active  = False
                prob                = 0.5

            # FPS counter
            dt = time.time() - t0
            fps_buf.append(1.0 / max(dt, 1e-6))
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            fps = np.mean(fps_buf)

            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # Chỉ báo điều kiện đang chờ để dễ debug
            status_str = {
                'NORMAL':   "Status: NORMAL",
                'WATCHING': f"Status: WATCHING ({elapsed:.2f}s / {drowsy_duration}s)",
                'ALERT':    f"Status: ALERT ({elapsed:.1f}s)",
            }[state]
            cv2.putText(display, status_str, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("ASIC Drowsiness Detection", display)

            elapsed_frame = time.time() - t0
            if elapsed_frame < frame_time:
                time.sleep(frame_time - elapsed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"capture_{int(time.time())}.png"
                cv2.imwrite(fname, frame)
                print(f"📸 Saved: {fname}")

            frame_i += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime drowsiness detection với INT8 TFLite model")
    parser.add_argument("--model",    default="drowsiness_asic_int8.tflite",
                        help="Đường dẫn tới .tflite model")
    parser.add_argument("--cam",      type=int, default=0,
                        help="Camera ID (mặc định 0)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DROWSY_DURATION,
                        help=f"Thời gian nhắm mắt tối thiểu để cảnh báo, đơn vị giây "
                             f"(mặc định {DEFAULT_DROWSY_DURATION}s)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Ẩn Q0.7 preprocessing preview")
    args = parser.parse_args()

    run_realtime(args.model, args.cam,
                 show_preprocessing=not args.no_preview,
                 drowsy_duration=args.duration)