"""
Realtime Webcam Detection — ASIC Driver Drowsiness
===================================================
Preprocessing hoàn toàn đồng bộ với pipeline offline:
  preprocess_data_asic.py  →  preprocessing_asic.py  →  config_asic.py

Pipeline 1 frame:
  1. Haar cascade → crop ROI khuôn mặt
  2. MediaPipe FaceMesh → landmark mắt trái + phải
  3. Crop từng mắt (padding 30%) → resize về 64×32 (INTER_AREA)
  4. Stack dọc [trái / phải] → (64, 64) uint8
  5. uint8 >> 1 → int8 Q0.7 [0, 127]   (đúng normalize_to_fixed_point)
  6. / 128.0 → float32 → INT8 TFLite inference
  7. Dequant → logit → sigmoid → prob → EMA + sliding window
  8. State machine 3 trạng thái: NORMAL / WATCHING / ALERT

Chạy:
  # Ngưỡng mặc định:
  python realtime_cam.py --model drowsiness_asic_int8.tflite

  # Ngưỡng từ Step 10 Colab (khuyến nghị):
  python realtime_cam.py --model drowsiness_asic_int8.tflite --calib thresholds.json

  python realtime_cam.py --model drowsiness_asic_int8.tflite --cam 0 --duration 2.0
"""

import cv2
import numpy as np
import argparse
import time
import sys
import json
import os
import csv
from collections import deque

# ── TFLite runtime ──────────────────────────────────────────────────────────
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
    mp_facemesh = mp.solutions.face_mesh
    mp_drawing  = mp.solutions.drawing_utils
    denorm      = mp_drawing._normalized_to_pixel_coordinates

    # Index landmark mắt — đồng bộ với preprocess_data_asic.py
    LEFT_EYE_IDXS  = set(np.ravel(list(mp_facemesh.FACEMESH_LEFT_EYE)))
    RIGHT_EYE_IDXS = set(np.ravel(list(mp_facemesh.FACEMESH_RIGHT_EYE)))
    MEDIAPIPE_OK    = True
    print("✅ MediaPipe FaceMesh ready")
except ImportError:
    MEDIAPIPE_OK = False
    print("❌ MediaPipe không có — không thể chạy (cần crop mắt)")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS — đồng bộ với config_asic.py + preprocess_data_asic.py
# ════════════════════════════════════════════════════════════════════════════

# config_asic.py
IMG_SIZE     = 64          # 2^6 × 2^6
SCALE_FACTOR = 128.0       # Q0.7: int8 / 128 → float32
QUANT_MAX    = 127         # int8 max

# preprocess_data_asic.py
EYE_W       = IMG_SIZE       # 64  — chiều rộng mỗi mắt
EYE_H       = IMG_SIZE // 2  # 32  — chiều cao mỗi mắt
EYE_PADDING = 0.30           # 30% padding mỗi phía (đúng với offline)

# ── Ngưỡng mặc định ─────────────────────────────────────────────────────────
# Sẽ bị ghi đè nếu dùng --calib thresholds.json (từ Step 10 Colab)
# prob = sigmoid(logit): 1.0 = mắt mở hoàn toàn, 0.0 = mắt nhắm hoàn toàn
FATIGUE_THRESHOLD  = 0.35   # prob < này → buồn ngủ (vùng nguy hiểm)
WATCHING_THRESHOLD = 0.55   # prob < này → bắt đầu theo dõi

# ── Smoothing ────────────────────────────────────────────────────────────────
EMA_ALPHA     = 0.50   # Exponential Moving Average: nhanh nhưng không giật
SMOOTH_WINDOW = 5      # Sliding window: trung bình 5 inference gần nhất

# ── Timing ───────────────────────────────────────────────────────────────────
DEFAULT_DROWSY_DURATION = 2.0   # Giây nhắm mắt liên tục → ALERT
TARGET_FPS              = 20
INFERENCE_STRIDE        = 2     # Chỉ inference 1/2 frame (tiết kiệm CPU)


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE PROFILER
# ════════════════════════════════════════════════════════════════════════════

class PipelineProfiler:
    """
    Đo thời gian chi tiết từng stage trong pipeline mỗi inference frame.

    7 stages:
      [A] cap.read()                  — đọc frame từ camera
      [B] Face detect (Haar)          — Haar cascade tìm khuôn mặt
      [C] FaceMesh landmark           — MediaPipe detect landmark mắt
      [D] Preprocess (gray+resize+Q0.7) — crop mắt, stack, quantize
      [E] TFLite inference            — INT8 model invoke
      [F] Dequant + sigmoid + smooth  — tính prob, EMA, sliding window
      [G] UI render + imshow          — vẽ UI lên frame và hiển thị

    In tóm tắt mỗi PRINT_EVERY inference frame.
    Xuất CSV khi kết thúc (--profile-csv).
    """

    STAGES = ['cap_read', 'face_detect', 'landmark', 'preprocess',
              'inference', 'postprocess', 'render']

    LABELS = {
        'cap_read'   : '[A] cap.read()',
        'face_detect': '[B] Face detect (Haar)',
        'landmark'   : '[C] FaceMesh landmark',
        'preprocess' : '[D] Preprocess (gray+resize+Q0.7)',
        'inference'  : '[E] TFLite inference',
        'postprocess': '[F] Dequant + sigmoid + smooth',
        'render'     : '[G] UI render + imshow',
    }

    # Nhóm "AI" = landmark + preprocess + inference + postprocess
    # Dùng để tính %AI trong tổng pipeline
    AI_STAGES = {'landmark', 'preprocess', 'inference', 'postprocess'}

    PRINT_EVERY = 30   # in tóm tắt mỗi N inference frame

    def __init__(self, csv_path='pipeline_timing.csv'):
        self.csv_path = csv_path
        self._t       = {}        # checkpoint timestamps
        self._sums    = {s: 0.0 for s in self.STAGES}
        self._count   = 0

        # Mở CSV ngay khi khởi tạo
        self._csv_file   = open(csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=['frame'] + self.STAGES + ['total_ms', 'ai_ms', 'ai_pct']
        )
        self._csv_writer.writeheader()
        print(f"\n⏱  PipelineProfiler — log: {csv_path}")
        print(f"   In tóm tắt mỗi {self.PRINT_EVERY} inference frame\n")

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def start(self):
        """Gọi đầu mỗi frame, trước cap.read()."""
        self._t = {'_start': time.perf_counter()}

    def mark(self, stage: str):
        """Gọi ngay SAU KHI stage kết thúc."""
        self._t[stage] = time.perf_counter()

    # ── Commit 1 inference frame ──────────────────────────────────────────

    def commit(self, frame_idx: int):
        """Tính delta, ghi CSV, cập nhật running stats."""
        keys = ['_start'] + self.STAGES
        if any(k not in self._t for k in keys):
            return   # frame thiếu checkpoint (bị skip inference)

        deltas = {}
        for i, stage in enumerate(self.STAGES):
            deltas[stage] = (self._t[stage] - self._t[keys[i]]) * 1000  # ms

        total_ms = sum(deltas.values())
        ai_ms    = sum(deltas[s] for s in self.AI_STAGES)
        ai_pct   = ai_ms / total_ms * 100 if total_ms > 0 else 0

        row = {'frame': frame_idx, **deltas,
               'total_ms': total_ms, 'ai_ms': ai_ms, 'ai_pct': ai_pct}
        self._csv_writer.writerow(
            {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in row.items()})
        self._csv_file.flush()

        for s in self.STAGES:
            self._sums[s] += deltas[s]
        self._count += 1

        if self._count % self.PRINT_EVERY == 0:
            self._print_summary(frame_idx, deltas, total_ms)

    # ── Summary printer ───────────────────────────────────────────────────

    def _print_summary(self, frame_idx, last_deltas, last_total):
        avg_total = sum(self._sums.values()) / self._count
        avg_ai    = sum(self._sums[s] for s in self.AI_STAGES) / self._count
        last_ai   = sum(last_deltas[s] for s in self.AI_STAGES)

        print(f"\n{'─'*66}")
        print(f"  ⏱  PIPELINE TIMING — frame #{frame_idx}  "
              f"(avg {self._count} inference frames)")
        print(f"{'─'*66}")
        print(f"  {'Stage':<38} {'Last':>8}  {'Avg':>8}  {'%Avg':>6}  Bar")
        print(f"  {'─'*38}  {'─'*8}  {'─'*8}  {'─'*6}")

        for s in self.STAGES:
            avg_s = self._sums[s] / self._count
            pct   = avg_s / avg_total * 100 if avg_total > 0 else 0
            bar   = '█' * max(1, int(pct / 3))
            # Đánh dấu AI stages bằng *
            tag   = ' *' if s in self.AI_STAGES else '  '
            print(f"  {self.LABELS[s]:<38}{tag} "
                  f"{last_deltas[s]:>7.2f}ms  "
                  f"{avg_s:>7.2f}ms  "
                  f"{pct:>5.1f}%  {bar}")

        print(f"  {'─'*38}  {'─'*8}  {'─'*8}  {'─'*6}")
        print(f"  {'TOTAL end-to-end':<40} "
              f"{last_total:>7.2f}ms  {avg_total:>7.2f}ms  100.0%")

        last_ai_pct = last_ai / last_total * 100 if last_total > 0 else 0
        avg_ai_pct  = avg_ai  / avg_total * 100 if avg_total > 0 else 0
        print(f"  {'AI stages (* C+D+E+F)':<40} "
              f"{last_ai:>7.2f}ms  {avg_ai:>7.2f}ms  "
              f"{avg_ai_pct:>5.1f}%  ← AI chiếm bao nhiêu")
        print(f"  Implied throughput: {1000/avg_total:.1f} inference/s\n")

    # ── Final summary + close ─────────────────────────────────────────────

    def close(self):
        if self._count == 0:
            self._csv_file.close()
            return

        avg_total = sum(self._sums.values()) / self._count
        avg_ai    = sum(self._sums[s] for s in self.AI_STAGES) / self._count

        print(f"\n{'═'*66}")
        print(f"  📊 FINAL PIPELINE SUMMARY  ({self._count} inference frames)")
        print(f"{'═'*66}")
        print(f"  {'Stage':<38} {'Mean':>8}  {'%':>6}  Bar")
        print(f"  {'─'*38}  {'─'*8}  {'─'*6}")

        for s in self.STAGES:
            avg_s = self._sums[s] / self._count
            pct   = avg_s / avg_total * 100 if avg_total > 0 else 0
            bar   = '█' * max(1, int(pct / 3))
            tag   = ' *' if s in self.AI_STAGES else '  '
            print(f"  {self.LABELS[s]:<38}{tag} {avg_s:>7.2f}ms  {pct:>5.1f}%  {bar}")

        print(f"  {'─'*38}  {'─'*8}  {'─'*6}")
        print(f"  {'TOTAL':<40} {avg_total:>7.2f}ms  100.0%")
        print(f"  {'AI (C+D+E+F) *':<40} {avg_ai:>7.2f}ms  "
              f"{avg_ai/avg_total*100:>5.1f}%  ← AI chiếm bao nhiêu")
        print(f"  Implied throughput  : {1000/avg_total:.1f} inference/s")
        print(f"  CSV đã lưu tại      : {self.csv_path}")
        print(f"{'═'*66}\n")

        self._csv_file.close()


# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_thresholds_from_json(json_path):
    """
    Đọc FATIGUE_THRESHOLD + WATCHING_THRESHOLD từ file JSON
    được tạo bởi Step 10 Colab (threshold_calibration.py).
    """
    if not os.path.isfile(json_path):
        print(f"⚠️  Không tìm thấy file calibration: {json_path}")
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        fat = float(data['FATIGUE_THRESHOLD'])
        wat = float(data['WATCHING_THRESHOLD'])

        print(f"✅ Đã đọc ngưỡng từ: {json_path}")
        print(f"   FATIGUE_THRESHOLD  = {fat}")
        print(f"   WATCHING_THRESHOLD = {wat}")
        extra = []
        for key in ('youden_j', 'tpr_at_threshold', 'fpr_at_threshold',
                    'prob_fatigue_mean', 'prob_active_mean', 'n_samples'):
            if key in data:
                extra.append(f"{key}={data[key]}")
        if extra:
            print(f"   Info: {', '.join(extra)}")
        return fat, wat
    except Exception as e:
        print(f"⚠️  Lỗi đọc calibration: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — đồng bộ chính xác với preprocess_data_asic.py
# ════════════════════════════════════════════════════════════════════════════

def _get_eye_bbox(landmarks, eye_idxs, imgW, imgH, padding=EYE_PADDING):
    """
    Tính bounding box mắt từ tập landmark indices.
    Đồng bộ hoàn toàn với _get_eye_bbox() trong preprocess_data_asic.py.
    """
    xs, ys = [], []
    for idx, lm in enumerate(landmarks):
        if idx not in eye_idxs:
            continue
        coord = denorm(lm.x, lm.y, imgW, imgH)
        if coord is None:
            continue
        xs.append(coord[0])
        ys.append(coord[1])

    if not xs:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Padding theo tỉ lệ bounding box — đúng với offline
    pw = int((x_max - x_min) * padding)
    ph = int((y_max - y_min) * padding)

    x1 = max(0,    x_min - pw)
    y1 = max(0,    y_min - ph)
    x2 = min(imgW, x_max + pw)
    y2 = min(imgH, y_max + ph)

    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def _crop_and_resize_eye(img_gray, bbox):
    """
    Crop vùng mắt → resize về EYE_W × EYE_H (64×32).
    Đồng bộ với _crop_and_prepare_eye() trong preprocess_data_asic.py.
    """
    x1, y1, x2, y2 = bbox
    crop = img_gray[y1:y2, x1:x2]
    # INTER_AREA: giống offline (tốt nhất khi downscale)
    return cv2.resize(crop, (EYE_W, EYE_H), interpolation=cv2.INTER_AREA)


def _normalize_to_fixed_point(img_uint8):
    """
    uint8 [0,255] → int8 Q0.7 [0,127]: value >> 1
    Đồng bộ với normalize_to_fixed_point() trong preprocessing_asic.py.
    """
    fixed = (img_uint8.astype(np.int16) >> 1).astype(np.int8)
    return np.clip(fixed, 0, QUANT_MAX)


def preprocess_roi(roi_bgr, face_mesh_ctx):
    """
    Chạy full preprocessing pipeline trên ROI khuôn mặt.
    Đồng bộ với crop_both_eyes_stacked() + process_with_eye_crop()
    trong preprocess_data_asic.py.

    Args:
        roi_bgr      : BGR image — ROI khuôn mặt từ Haar cascade
        face_mesh_ctx: MediaPipe FaceMesh instance (persistent, không tạo mới mỗi frame)

    Returns:
        fp32_input : float32 (1, 64, 64, 1) — sẵn sàng đưa vào TFLite
        q07        : int8   (64, 64)        — để preview Q0.7
        bbox_left  : tuple hoặc None
        bbox_right : tuple hoặc None
    """
    roi_bgr = np.ascontiguousarray(roi_bgr)
    imgH, imgW = roi_bgr.shape[:2]

    # Bước 1: Grayscale (đúng với offline — dùng COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Bước 2: MediaPipe FaceMesh — dùng instance đã khởi tạo bên ngoài
    # (static_image_mode=False → dùng tracking, nhanh hơn nhiều sau frame đầu)
    results = face_mesh_ctx.process(roi_bgr)

    if not results.multi_face_landmarks:
        return None, None, None, None

    lms = results.multi_face_landmarks[0].landmark

    # Bước 3: Lấy bbox từng mắt
    bbox_left  = _get_eye_bbox(lms, LEFT_EYE_IDXS,  imgW, imgH)
    bbox_right = _get_eye_bbox(lms, RIGHT_EYE_IDXS, imgW, imgH)

    if bbox_left is None or bbox_right is None:
        return None, None, None, None

    # Bước 4: Crop + resize từng mắt → 64×32
    eye_left  = _crop_and_resize_eye(img_gray, bbox_left)   # (32, 64) uint8
    eye_right = _crop_and_resize_eye(img_gray, bbox_right)  # (32, 64) uint8

    # Bước 5: Stack dọc → (64, 64) uint8
    stacked = np.vstack([eye_left, eye_right])

    # Bước 6: uint8 >> 1 → int8 Q0.7 [0, 127]
    q07 = _normalize_to_fixed_point(stacked)

    # Bước 7: int8 / 128.0 → float32, add batch+channel dim
    fp32 = q07.astype(np.float32) / SCALE_FACTOR
    fp32_input = fp32[np.newaxis, :, :, np.newaxis]  # (1, 64, 64, 1)

    return fp32_input, q07, bbox_left, bbox_right


# ════════════════════════════════════════════════════════════════════════════
# TFLITE INFERENCE
# ════════════════════════════════════════════════════════════════════════════

class DrowsinessDetector:
    """Wrap TFLite INT8 interpreter + smoothing."""

    def __init__(self, model_path):
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()

        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]

        self.inp_idx   = inp['index']
        self.out_idx   = out['index']
        self.inp_scale = float(inp['quantization_parameters']['scales'][0])
        self.inp_zp    = int(inp['quantization_parameters']['zero_points'][0])
        self.out_scale = float(out['quantization_parameters']['scales'][0])
        self.out_zp    = int(out['quantization_parameters']['zero_points'][0])
        self.inp_dtype = inp['dtype']

        # Smoothing state
        self._ema    = 0.5
        self._window = deque(maxlen=SMOOTH_WINDOW)

        print(f"\n🔢 TFLite quantization params:")
        print(f"   Input  scale={self.inp_scale:.8f}, zp={self.inp_zp}, dtype={self.inp_dtype}")
        print(f"   Output scale={self.out_scale:.8f}, zp={self.out_zp}")

    def predict(self, fp32_input):
        """
        Args:
            fp32_input: float32 (1, 64, 64, 1) — output của preprocess_roi()

        Returns:
            prob_raw   : float — prob từ model (chưa smooth)
            prob_smooth: float — EMA + sliding window
            logit      : float — raw logit trước sigmoid
        """
        # Quantize float32 → int8 theo scale/zp của model
        inp_int8 = np.round(
            fp32_input / self.inp_scale + self.inp_zp
        ).clip(-128, 127).astype(np.int8)

        self.interp.set_tensor(self.inp_idx, inp_int8)
        self.interp.invoke()

        out      = self.interp.get_tensor(self.out_idx)
        logit    = float((int(out[0][0]) - self.out_zp) * self.out_scale)
        prob_raw = float(1.0 / (1.0 + np.exp(-logit)))

        # EMA
        self._ema = EMA_ALPHA * prob_raw + (1.0 - EMA_ALPHA) * self._ema

        # Sliding window
        self._window.append(prob_raw)
        prob_smooth = float(np.mean(self._window))

        return prob_raw, prob_smooth, logit

    def reset(self):
        self._window.clear()
        self._ema = 0.5


# ════════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ════════════════════════════════════════════════════════════════════════════

class DrowsinessGuard:
    """
    3 trạng thái dựa trên prob_smooth:

      NORMAL   — prob_smooth >= WATCHING_THRESHOLD  (mắt mở)
      WATCHING — FATIGUE_THRESHOLD <= prob_smooth < WATCHING_THRESHOLD
      ALERT    — prob_smooth < FATIGUE_THRESHOLD liên tục >= min_duration

    ALERT chỉ trigger từ vùng FATIGUE (không trigger từ WATCHING),
    tránh false positive do nháy mắt hoặc nghiêng đầu.
    """

    def __init__(self, min_duration=DEFAULT_DROWSY_DURATION):
        self.min_duration  = min_duration
        self.fatigue_since = None
        self.alert_active  = False

    def update(self, prob_smooth):
        now = time.time()

        if prob_smooth < FATIGUE_THRESHOLD:
            if self.fatigue_since is None:
                self.fatigue_since = now
            elapsed = now - self.fatigue_since
            if elapsed >= self.min_duration:
                self.alert_active = True
                return 'ALERT', elapsed
            return 'WATCHING', elapsed

        elif prob_smooth < WATCHING_THRESHOLD:
            if self.fatigue_since is None:
                self.fatigue_since = now
            elapsed = now - self.fatigue_since
            return 'WATCHING', elapsed

        else:
            self.fatigue_since = None
            self.alert_active  = False
            return 'NORMAL', 0.0

    def reset(self):
        self.fatigue_since = None
        self.alert_active  = False


# ════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _draw_eye_boxes(display, roi_offset, bbox_left, bbox_right):
    """Vẽ bbox mắt trái + phải lên frame gốc (offset theo vị trí ROI)."""
    ox, oy = roi_offset
    for bbox, color in [(bbox_left, (255, 200, 0)), (bbox_right, (0, 200, 255))]:
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display,
                      (ox + x1, oy + y1), (ox + x2, oy + y2),
                      color, 1)


def _draw_eye_preview(display, q07):
    """
    Vẽ preview Q0.7 ảnh mắt (64×64 int8) góc trên phải.
    Hiển thị đúng dạng: int8 × 2 → uint8 (đảo ngược normalize_to_fixed_point).
    """
    if q07 is None:
        return
    # Denormalize: int8 << 1 → uint8 (đồng bộ denormalize_from_fixed_point)
    prev_u8  = np.clip(q07.astype(np.int16) * 2, 0, 255).astype(np.uint8)
    prev_bgr = cv2.cvtColor(prev_u8, cv2.COLOR_GRAY2BGR)
    prev_bgr = cv2.resize(prev_bgr, (IMG_SIZE, IMG_SIZE))

    x1 = display.shape[1] - IMG_SIZE - 10
    y1 = 10
    display[y1:y1+IMG_SIZE, x1:x1+IMG_SIZE] = prev_bgr

    # Đường phân cách giữa mắt trái và phải (hàng 32)
    mid_y = y1 + EYE_H
    cv2.line(display, (x1, mid_y), (x1+IMG_SIZE, mid_y), (80, 80, 80), 1)

    cv2.putText(display, "L", (x1 + 2, y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
    cv2.putText(display, "R", (x1 + 2, mid_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
    cv2.putText(display, "Q0.7 preview",
                (x1, y1 + IMG_SIZE + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)


def _draw_prob_bar(display, prob_sm, drowsy_duration):
    """Thanh prob_smooth ở cuối màn hình với markers ngưỡng."""
    H, W = display.shape[:2]
    bx1, by1 = 10,     H - 30
    bx2, by2 = W - 10, H - 18
    bar_w = bx2 - bx1

    # Nền
    cv2.rectangle(display, (bx1, by1), (bx2, by2), (50, 50, 50), -1)

    # Fill theo prob
    fill = int(bar_w * np.clip(prob_sm, 0, 1))
    color = (0, 220, 0)   if prob_sm >= WATCHING_THRESHOLD else \
            (0, 215, 255) if prob_sm >= FATIGUE_THRESHOLD  else \
            (0, 0, 255)
    cv2.rectangle(display, (bx1, by1), (bx1 + fill, by2), color, -1)

    # Marker ngưỡng
    for thr, col in [(WATCHING_THRESHOLD, (0, 215, 255)),
                     (FATIGUE_THRESHOLD,  (0, 0, 255))]:
        mx = bx1 + int(bar_w * thr)
        cv2.line(display, (mx, by1), (mx, by2), col, 2)

    cv2.putText(display, f"prob={prob_sm:.3f}",
                (bx1, by1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def _draw_face_state(display, x, y, w, h,
                     state, prob_sm, elapsed, drowsy_duration):
    """Vẽ bounding box khuôn mặt + label theo trạng thái."""

    if state == 'ALERT':
        # Overlay đỏ toàn màn hình
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0),
                      (display.shape[1], display.shape[0]),
                      (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.30, display, 0.70, 0, display)

        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(display,
                    f"MAT NHAM! prob={prob_sm:.2f} ({elapsed:.1f}s)",
                    (x, y - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.putText(display, "!! CANH BAO: HAY DUNG XE !!",
                    (display.shape[1]//2 - 210, display.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    elif state == 'WATCHING':
        col = (0, 100, 255) if prob_sm < FATIGUE_THRESHOLD else (0, 215, 255)
        cv2.rectangle(display, (x, y), (x+w, y+h), col, 2)
        cv2.putText(display,
                    f"WATCHING prob={prob_sm:.2f} ({elapsed:.1f}s/{drowsy_duration:.1f}s)",
                    (x, y - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 2)
        # Progress bar dưới bbox mặt
        progress = min(elapsed / drowsy_duration, 1.0)
        bar_w = int(w * progress)
        cv2.rectangle(display, (x, y+h+4), (x+bar_w, y+h+14), col, -1)
        cv2.rectangle(display, (x, y+h+4), (x+w, y+h+14), (80, 80, 80), 1)

    else:  # NORMAL
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 0), 2)
        cv2.putText(display,
                    f"ACTIVE prob={prob_sm:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 220, 0), 2)


# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_realtime(model_path, cam_id=0, show_preview=True,
                 drowsy_duration=DEFAULT_DROWSY_DURATION,
                 calib_json=None, profile_csv='pipeline_timing.csv'):

    # ── Load ngưỡng calibration nếu có ───────────────────────────────────
    global FATIGUE_THRESHOLD, WATCHING_THRESHOLD
    if calib_json is not None:
        result = load_thresholds_from_json(calib_json)
        if result is not None:
            FATIGUE_THRESHOLD, WATCHING_THRESHOLD = result
        else:
            print(f"⚠️  Dùng ngưỡng mặc định: "
                  f"FATIGUE={FATIGUE_THRESHOLD}, WATCHING={WATCHING_THRESHOLD}")
    else:
        print(f"ℹ️  Ngưỡng mặc định: "
              f"FATIGUE={FATIGUE_THRESHOLD}, WATCHING={WATCHING_THRESHOLD}")
        print(f"   Tip: Chạy step10_threshold_calibration.py trên Colab")
        print(f"        rồi dùng --calib thresholds.json để có ngưỡng chính xác\n")

    # ── Khởi tạo ──────────────────────────────────────────────────────────
    detector = DrowsinessDetector(model_path)
    guard    = DrowsinessGuard(min_duration=drowsy_duration)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Không load được Haar cascade")
        sys.exit(1)

    # MediaPipe FaceMesh persistent — static_image_mode=False để dùng tracking
    # Nhanh hơn ~5x so với tạo mới mỗi frame (static_image_mode=True)
    face_mesh = mp_facemesh.FaceMesh(
        static_image_mode=False,      # tracking mode: dùng lại landmark từ frame trước
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,  # tracking confidence — chỉ có khi mode=False
    )
    if face_cascade.empty():
        print("❌ Không load được Haar cascade")
        sys.exit(1)

    # Profiler — đo thời gian từng stage
    profiler = PipelineProfiler(csv_path=profile_csv)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"❌ Không mở được camera {cam_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)

    frame_time = 1.0 / TARGET_FPS

    print(f"\n{'='*60}")
    print(f"🎥 Camera đang chạy — Q: thoát  |  S: chụp ảnh")
    print(f"{'='*60}")
    print(f"  FPS target        : {TARGET_FPS}")
    print(f"  Inference stride  : mỗi {INFERENCE_STRIDE} frame")
    print(f"  FATIGUE_THRESHOLD : prob < {FATIGUE_THRESHOLD:.3f}")
    print(f"  WATCHING_THRESHOLD: prob < {WATCHING_THRESHOLD:.3f}")
    print(f"  Alert khi          : prob < {FATIGUE_THRESHOLD:.3f}  liên tục ≥ {drowsy_duration}s")
    print(f"  Eye layout         : hàng  0–{EYE_H-1} = mắt trái")
    print(f"                       hàng {EYE_H}–{IMG_SIZE-1} = mắt phải")
    print(f"{'='*60}\n")

    fps_buf    = deque(maxlen=30)
    prob_raw   = 0.5
    prob_sm    = 0.5
    logit      = 0.0
    state      = 'NORMAL'
    elapsed    = 0.0
    frame_i    = 0
    q07        = None
    bbox_left  = None
    bbox_right = None

    try:
        while True:
            # ── [A] cap.read ─────────────────────────────────────────────
            profiler.start()
            t0  = time.time()
            ret, frame = cap.read()
            profiler.mark('cap_read')
            if not ret:
                print("❌ Không đọc được frame")
                break

            display  = frame.copy()
            gray_fr  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── [B] Haar cascade detect khuôn mặt ────────────────────────
            # scaleFactor=1.1: quét tỉ mỉ hơn (1.3 bỏ sót nhiều góc nghiêng)
            # minNeighbors=3 : bớt chặt hơn, giảm miss detection
            # minSize=(80,80): bỏ qua khuôn mặt quá nhỏ / nhiễu background
            faces = face_cascade.detectMultiScale(
                gray_fr,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            profiler.mark('face_detect')

            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = frame[y:y+h, x:x+w]

                # ── Inference mỗi INFERENCE_STRIDE frame ───────────────
                if frame_i % INFERENCE_STRIDE == 0:

                    # ── [C] FaceMesh landmark (bên trong preprocess_roi) ──
                    roi_c = np.ascontiguousarray(roi)
                    img_gray_roi = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
                    results      = face_mesh.process(roi_c)
                    profiler.mark('landmark')

                    # ── [D] Preprocess: crop mắt + stack + Q0.7 ──────────
                    if results.multi_face_landmarks:
                        lms    = results.multi_face_landmarks[0].landmark
                        imgH_r, imgW_r = roi_c.shape[:2]
                        bl = _get_eye_bbox(lms, LEFT_EYE_IDXS,  imgW_r, imgH_r)
                        br = _get_eye_bbox(lms, RIGHT_EYE_IDXS, imgW_r, imgH_r)

                        if bl is not None and br is not None:
                            eye_l   = _crop_and_resize_eye(img_gray_roi, bl)
                            eye_r   = _crop_and_resize_eye(img_gray_roi, br)
                            stacked = np.vstack([eye_l, eye_r])
                            q07     = _normalize_to_fixed_point(stacked)
                            fp32_inp = (q07.astype(np.float32) / SCALE_FACTOR
                                        )[np.newaxis, :, :, np.newaxis]
                            bbox_left, bbox_right = bl, br
                        else:
                            fp32_inp = None
                            q07 = bbox_left = bbox_right = None
                    else:
                        fp32_inp = None
                        q07 = bbox_left = bbox_right = None
                    profiler.mark('preprocess')

                    # ── [E] TFLite inference ──────────────────────────────
                    if fp32_inp is not None:
                        inp_int8 = np.round(
                            fp32_inp / detector.inp_scale + detector.inp_zp
                        ).clip(-128, 127).astype(np.int8)
                        detector.interp.set_tensor(detector.inp_idx, inp_int8)
                        detector.interp.invoke()
                        profiler.mark('inference')

                        # ── [F] Dequant + sigmoid + smoothing ─────────────
                        out    = detector.interp.get_tensor(detector.out_idx)
                        logit  = float((int(out[0][0]) - detector.out_zp)
                                       * detector.out_scale)
                        prob_raw = float(1.0 / (1.0 + np.exp(-logit)))
                        detector._ema = (EMA_ALPHA * prob_raw
                                         + (1 - EMA_ALPHA) * detector._ema)
                        detector._window.append(prob_raw)
                        prob_sm = float(np.mean(detector._window))
                        profiler.mark('postprocess')

                        print(f"[f{frame_i:5d}] logit={logit:+.3f}  "
                              f"raw={prob_raw:.3f}  smooth={prob_sm:.3f}  "
                              f"state={state}")
                    else:
                        # Không có mắt — điền placeholder để profiler không lệch
                        profiler.mark('inference')
                        profiler.mark('postprocess')
                        print(f"[f{frame_i:5d}] ⚠️  No eye landmarks detected")

                else:
                    # Frame skip — điền placeholder cho đủ 7 checkpoint
                    profiler.mark('landmark')
                    profiler.mark('preprocess')
                    profiler.mark('inference')
                    profiler.mark('postprocess')

                # ── Cập nhật state machine ──────────────────────────────
                state, elapsed = guard.update(prob_sm)

                # ── Vẽ UI ───────────────────────────────────────────────
                _draw_face_state(display, x, y, w, h,
                                 state, prob_sm, elapsed, drowsy_duration)
                if bbox_left or bbox_right:
                    _draw_eye_boxes(display, (x, y), bbox_left, bbox_right)
                if show_preview:
                    _draw_eye_preview(display, q07)

            else:
                # Không thấy mặt → reset + placeholder landmarks
                profiler.mark('landmark')
                profiler.mark('preprocess')
                profiler.mark('inference')
                profiler.mark('postprocess')

                cv2.putText(display, "No face detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                guard.reset()
                detector.reset()
                prob_sm   = 0.5
                q07       = None
                bbox_left = bbox_right = None

            # ── [G] FPS + prob bar + UI render ───────────────────────────
            dt = time.time() - t0
            fps_buf.append(1.0 / max(dt, 1e-6))
            fps = float(np.mean(fps_buf))

            _draw_prob_bar(display, prob_sm, drowsy_duration)

            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            status_str = {
                'NORMAL':   f"NORMAL   prob={prob_sm:.3f}",
                'WATCHING': f"WATCHING prob={prob_sm:.3f}  {elapsed:.1f}s/{drowsy_duration:.1f}s",
                'ALERT':    f"ALERT    prob={prob_sm:.3f}  {elapsed:.1f}s",
            }[state]
            cv2.putText(display, status_str, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.putText(display,
                        f"THR fatigue={FATIGUE_THRESHOLD:.2f}  watching={WATCHING_THRESHOLD:.2f}",
                        (10, display.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)

            cv2.imshow("ASIC Drowsiness Detection", display)
            profiler.mark('render')

            # Chỉ commit inference frame (có đủ 7 checkpoint có nghĩa)
            if frame_i % INFERENCE_STRIDE == 0 and len(faces) > 0:
                profiler.commit(frame_i)

            # Giới hạn FPS
            elapsed_frame = time.time() - t0
            if elapsed_frame < frame_time:
                time.sleep(frame_time - elapsed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Thoát.")
                break
            elif key == ord('s'):
                fname = f"capture_{int(time.time())}.png"
                cv2.imwrite(fname, frame)
                print(f"📸 Saved: {fname}")

            frame_i += 1

    finally:
        cap.release()
        face_mesh.close()
        profiler.close()
        cv2.destroyAllWindows()
        print("✅ Camera closed.")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime Drowsiness Detection — ASIC pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", default="drowsiness_asic_int8.tflite",
        help="Đường dẫn file TFLite INT8 (mặc định: drowsiness_asic_int8.tflite)",
    )
    parser.add_argument(
        "--cam", type=int, default=0,
        help="Camera index (mặc định: 0)",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DROWSY_DURATION,
        help=f"Giây nhắm mắt liên tục để cảnh báo (mặc định: {DEFAULT_DROWSY_DURATION}s)",
    )
    parser.add_argument(
        "--calib", default=None, metavar="thresholds.json",
        help=(
            "File JSON từ step10_threshold_calibration.py trên Colab.\n"
            "Chứa FATIGUE_THRESHOLD và WATCHING_THRESHOLD đã calibrate.\n"
            "Ví dụ: --calib thresholds.json\n"
            "Nếu không cung cấp: dùng ngưỡng mặc định hardcode."
        ),
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Tắt preview ảnh mắt Q0.7 góc trên phải",
    )
    parser.add_argument(
        "--profile-csv", default="pipeline_timing.csv",
        metavar="FILE.csv",
        help="Đường dẫn file CSV xuất kết quả profiling (mặc định: pipeline_timing.csv)",
    )
    args = parser.parse_args()

    run_realtime(
        model_path      = args.model,
        cam_id          = args.cam,
        show_preview    = not args.no_preview,
        drowsy_duration = args.duration,
        calib_json      = args.calib,
        profile_csv     = args.profile_csv,
    )