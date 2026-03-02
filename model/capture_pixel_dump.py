"""
MODULE 2 — Chụp ảnh & Xuất thông tin pixel trước khi vào model
================================================================
Chụp 1 ảnh từ camera (hoặc load file), thực hiện TOÀN BỘ preprocessing,
rồi xuất ra:
  - Ảnh gốc
  - Ảnh sau grayscale + resize
  - Ảnh Q0.7 int8 (chính xác như ASIC nhận)
  - File TXT chứa từng pixel value (để so sánh với RTL testbench)
  - File CSV chứa pixel theo format: row,col,uint8_raw,int8_q07,float32

Chạy:
  python capture_pixel_dump.py --model drowsiness_asic_int8.tflite
  python capture_pixel_dump.py --image my_photo.jpg   # dùng ảnh có sẵn
  python capture_pixel_dump.py --cam 0                # chụp từ webcam
"""

import cv2
import numpy as np
import argparse
import time
import os

try:
    import mediapipe as mp
    mp_facemesh  = mp.solutions.face_mesh
    mp_drawing   = mp.solutions.drawing_utils
    denorm_coord = mp_drawing._normalized_to_pixel_coordinates
    MEDIAPIPE_OK = True
    _left  = set(np.ravel(list(mp_facemesh.FACEMESH_LEFT_EYE)))
    _right = set(np.ravel(list(mp_facemesh.FACEMESH_RIGHT_EYE)))
    EYE_IDXS = _left | _right
except ImportError:
    MEDIAPIPE_OK = False
    print("⚠️  MediaPipe không có")

IMG_SIZE     = 128
SCALE_FACTOR = 128.0


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (bản sao chính xác từ pipeline training)
# ════════════════════════════════════════════════════════════════════════════

def draw_landmarks(roi_bgr):
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
                    image=out, landmark_list=fl,
                    connections=mp_facemesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=conn_spec)
                for idx, lm in enumerate(fl.landmark):
                    if idx in EYE_IDXS:
                        pt = denorm_coord(lm.x, lm.y, w, h)
                        if pt:
                            cv2.circle(out, pt, 3, (255, 255, 255), -1)
    return out


def full_preprocess(roi_bgr, verbose=True):
    """
    Thực hiện preprocessing đầy đủ và trả về từng bước trung gian.

    Returns:
        dict với keys: '01_roi_bgr', '02_with_landmarks', '03_grayscale',
                       '04_resized_uint8', '05_q07_int8', '06_float32'
    """
    steps = {}

    # B1: Ảnh gốc ROI
    steps['01_roi_bgr'] = roi_bgr.copy()
    if verbose:
        print(f"  [1] ROI gốc          : shape={roi_bgr.shape}, dtype={roi_bgr.dtype}")

    # B2: Vẽ MediaPipe landmarks
    with_lm = draw_landmarks(roi_bgr)
    steps['02_with_landmarks'] = with_lm
    if verbose:
        print(f"  [2] Sau landmarks    : shape={with_lm.shape}")

    # B3: Grayscale
    gray = cv2.cvtColor(with_lm, cv2.COLOR_BGR2GRAY)
    steps['03_grayscale'] = gray
    if verbose:
        print(f"  [3] Grayscale        : shape={gray.shape}, dtype={gray.dtype}")
        print(f"       range=[{gray.min()}, {gray.max()}]")

    # B4: Resize 128×128
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    steps['04_resized_uint8'] = resized
    if verbose:
        print(f"  [4] Resized uint8    : shape={resized.shape}")
        print(f"       range=[{resized.min()}, {resized.max()}]")

    # B5: >> 1  →  Q0.7 int8 [0,127]
    q07 = np.clip((resized.astype(np.int16) >> 1), 0, 127).astype(np.int8)
    steps['05_q07_int8'] = q07
    if verbose:
        print(f"  [5] Q0.7 int8        : dtype={q07.dtype}")
        print(f"       range=[{q07.min()}, {q07.max()}]")
        print(f"       mean={q07.astype(float).mean():.2f}, "
              f"std={q07.astype(float).std():.2f}")
        print(f"       → Đây là dữ liệu INPUT CHÍNH XÁC đưa vào ASIC IP core")

    # B6: float32 cho TFLite
    fp32 = q07.astype(np.float32) / SCALE_FACTOR
    steps['06_float32'] = fp32
    if verbose:
        print(f"  [6] float32          : range=[{fp32.min():.4f}, {fp32.max():.4f}]")

    return steps


# ════════════════════════════════════════════════════════════════════════════
# CAPTURE FROM CAMERA OR FILE
# ════════════════════════════════════════════════════════════════════════════

def capture_from_cam(cam_id=0):
    """Hiện live preview, nhấn SPACE để chụp"""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera {cam_id}")

    print("📷 Nhấn SPACE để chụp, Q để thoát")
    frame = None
    try:
        while True:
            ret, f = cap.read()
            if not ret:
                break
            cv2.putText(f, "SPACE = chup | Q = thoat", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Capture — SPACE to shoot", f)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                frame = f.copy()
                break
            elif k == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return frame


def get_face_roi(frame):
    """
    Detect face và trả về ROI, cùng tọa độ.

    Returns:
        roi       : np.ndarray — vùng ảnh khuôn mặt (hoặc toàn frame nếu không detect)
        face_rect : tuple (x,y,w,h) hoặc None
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("⚠️  Không detect được face — dùng toàn bộ frame")
        return frame, None

    x, y, w, h = faces[0]
    print(f"  Face detected: x={x}, y={y}, w={w}, h={h}")
    return frame[y:y+h, x:x+w], (x, y, w, h)


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT / DUMP
# ════════════════════════════════════════════════════════════════════════════

def save_all_outputs(steps, out_dir, timestamp):
    """Lưu tất cả ảnh trung gian và file txt/csv"""
    os.makedirs(out_dir, exist_ok=True)

    # ── Lưu ảnh từng bước ───────────────────────────────────────────────
    for key, img in steps.items():
        if img.dtype == np.int8:
            # Q0.7 max=127 → nhân 2 để fill range uint8
            disp = np.clip(img.astype(np.int16) * 2, 0, 255).astype(np.uint8)
        elif img.dtype == np.float32:
            disp = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            disp = img

        if len(disp.shape) == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        path = os.path.join(out_dir, f"{timestamp}_{key}.png")
        cv2.imwrite(path, disp)
        print(f"  💾 {path}")

    # ── FILE TXT: pixel values dạng hex (cho RTL testbench) ─────────────
    q07 = steps['05_q07_int8']

    txt_hex_path = os.path.join(out_dir, f"{timestamp}_pixels_hex.txt")
    with open(txt_hex_path, 'w') as f:
        f.write("// ASIC Input Pixel Dump — Q0.7 int8 format\n")
        f.write(f"// Image size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write("// Format: 1 pixel per line, hexadecimal (8-bit)\n")
        f.write("// Row-major order: pixel[0] = row=0, col=0\n")
        f.write(f"// Total pixels: {IMG_SIZE * IMG_SIZE}\n\n")
        for row in range(IMG_SIZE):
            for col in range(IMG_SIZE):
                val = int(q07[row, col])
                f.write(f"{val & 0xFF:02X}\n")

    print(f"  💾 {txt_hex_path}  ← dùng cho RTL testbench (hex, 1 giá trị/dòng)")

    # ── FILE TXT: pixel values dạng decimal ─────────────────────────────
    txt_dec_path = os.path.join(out_dir, f"{timestamp}_pixels_dec.txt")
    with open(txt_dec_path, 'w') as f:
        f.write("// ASIC Input Pixel Dump — Decimal\n")
        f.write(f"// Size: {IMG_SIZE}x{IMG_SIZE}, dtype: int8, range: [0,127]\n\n")
        for row in range(IMG_SIZE):
            for col in range(IMG_SIZE):
                f.write(f"{int(q07[row, col])}\n")

    print(f"  💾 {txt_dec_path}")

    # ── FILE TXT: dạng 2D matrix (dễ đọc bằng mắt) ──────────────────────
    txt_matrix_path = os.path.join(out_dir, f"{timestamp}_pixels_matrix.txt")
    with open(txt_matrix_path, 'w') as f:
        f.write(f"// ASIC Input — 2D Matrix view ({IMG_SIZE}x{IMG_SIZE})\n")
        f.write("// dtype: int8, range: [0, 127]\n\n")
        for row in range(IMG_SIZE):
            row_str = " ".join(f"{int(q07[row, col]):3d}" for col in range(IMG_SIZE))
            f.write(row_str + "\n")

    print(f"  💾 {txt_matrix_path}  ← 2D matrix, dễ đọc")

    # ── FILE CSV: đầy đủ thông tin từng pixel ───────────────────────────
    # FIX: addr tính đúng cho mọi IMG_SIZE, không hardcode shift
    csv_path  = os.path.join(out_dir, f"{timestamp}_pixels_full.csv")
    uint8_img = steps['04_resized_uint8']
    fp32_img  = steps['06_float32']
    with open(csv_path, 'w') as f:
        f.write("row,col,addr,uint8_raw,int8_q07,hex_q07,float32\n")
        for row in range(IMG_SIZE):
            for col in range(IMG_SIZE):
                addr = row * IMG_SIZE + col          # row-major linear address
                u8   = int(uint8_img[row, col])
                i8   = int(q07[row, col])
                fp   = float(fp32_img[row, col])
                f.write(f"{row},{col},{addr},{u8},{i8},{i8 & 0xFF:02X},{fp:.6f}\n")

    print(f"  💾 {csv_path}  ← CSV đầy đủ (row,col,addr,uint8,int8,hex,float32)")

    # ── Thống kê tóm tắt ─────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, f"{timestamp}_summary.txt")
    with open(summary_path, 'w') as f:
        q = q07.astype(np.float64)
        f.write("=" * 50 + "\n")
        f.write("PIXEL STATISTICS (Q0.7 int8)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Image size  : {IMG_SIZE} x {IMG_SIZE}\n")
        f.write(f"Total pixels: {IMG_SIZE * IMG_SIZE}\n")
        f.write(f"dtype       : int8\n")
        f.write(f"Range       : [{int(q07.min())}, {int(q07.max())}]\n")
        f.write(f"Mean        : {q.mean():.4f}\n")
        f.write(f"Std         : {q.std():.4f}\n")
        f.write(f"Median      : {np.median(q):.4f}\n")
        f.write(f"\nHistogram (bins):\n")
        for lo, hi in [(0, 31), (32, 63), (64, 95), (96, 127)]:
            cnt = int(((q07 >= lo) & (q07 <= hi)).sum())
            f.write(f"  [{lo:3d}-{hi:3d}]: {cnt:6d}  "
                    f"({100 * cnt / (IMG_SIZE ** 2):.1f}%)\n")
        # Sample center 5×5
        cy = IMG_SIZE // 2
        cx = IMG_SIZE // 2
        f.write(f"\nSample pixels (center 5x5, row {cy-2}–{cy+2}, "
                f"col {cx-2}–{cx+2}):\n")
        for r in range(cy - 2, cy + 3):
            row_s = " ".join(f"{int(q07[r, c]):3d}"
                             for c in range(cx - 2, cx + 3))
            f.write(f"  row[{r}]: {row_s}\n")

    print(f"  💾 {summary_path}  ← thống kê")
    print(f"\n  ✅ Tất cả output lưu trong: {out_dir}/")

    return {
        'hex':     txt_hex_path,
        'dec':     txt_dec_path,
        'matrix':  txt_matrix_path,
        'csv':     csv_path,
        'summary': summary_path,
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Chụp ảnh + dump pixel info trước khi đưa vào ASIC model")
    parser.add_argument("--image", default=None,
                        help="Dùng file ảnh có sẵn thay vì camera")
    parser.add_argument("--cam",   type=int, default=0,
                        help="Camera ID (mặc định 0)")
    parser.add_argument("--out",   default="pixel_dumps",
                        help="Thư mục output (mặc định: pixel_dumps/)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("MODULE 2 — PIXEL DUMP TRƯỚC KHI VÀO MODEL")
    print("=" * 60)

    # Lấy frame
    if args.image:
        print(f"\n📂 Load ảnh: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Không đọc được: {args.image}")
    else:
        print(f"\n📷 Chụp từ camera {args.cam}...")
        frame = capture_from_cam(args.cam)
        if frame is None:
            print("❌ Không chụp được ảnh")
            return

    # Lưu ảnh gốc — tạo thư mục trước
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, f"{timestamp}_00_original.png")
    cv2.imwrite(raw_path, frame)
    print(f"  💾 Ảnh gốc: {raw_path}")

    # Detect face
    print(f"\n🔍 Detecting face...")
    roi, face_rect = get_face_roi(frame)

    # Full preprocessing
    print(f"\n⚙️  Preprocessing pipeline:")
    steps = full_preprocess(roi, verbose=True)

    # Lưu tất cả outputs
    print(f"\n💾 Lưu output files:")
    output_files = save_all_outputs(steps, args.out, timestamp)

    print(f"\n{'=' * 60}")
    print(f"📋 FILE QUAN TRỌNG CHO RTL TESTBENCH:")
    print(f"{'=' * 60}")
    print(f"  • {output_files['hex']}     ← input cho RTL ($readmemh)")
    print(f"  • {output_files['dec']}     ← input cho RTL (decimal)")
    print(f"  • {output_files['csv']}     ← đầy đủ thông tin")
    print(f"  • {output_files['summary']} ← thống kê")
    print(f"\n⚙️  Trong RTL testbench, đọc file hex như sau:")
    hex_base = os.path.basename(output_files['hex'])
    print(f"  $readmemh(\"{hex_base}\", pixel_mem);")
    print(f"  // pixel_mem là array [0:{IMG_SIZE*IMG_SIZE-1}] of [7:0]"
          f"  ({IMG_SIZE}×{IMG_SIZE}={IMG_SIZE*IMG_SIZE} pixels)")


if __name__ == "__main__":
    main()