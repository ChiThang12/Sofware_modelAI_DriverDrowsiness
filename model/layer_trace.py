"""
MODULE 4 — Layer-by-layer Inference Trace cho RTL Comparison
=============================================================
Lấy input từ Module 2 (Q0.7 int8 pixel dump), đưa qua từng layer,
và in/lưu output của TỪNG LAYER dưới dạng integer.

Mục đích: So sánh với output của RTL testbench để verify ASIC implementation.

Pipeline trace:
  Input int8 (từ file Module 2)
  → Conv1 → BN_folded → ReLU → MaxPool  → dump layer1_output.txt
  → Conv2 → BN_folded → ReLU → MaxPool  → dump layer2_output.txt
  → ...
  → GlobalAvgPool                         → dump gap_output.txt
  → Dense1 → ReLU                         → dump fc1_output.txt
  → Dense2 (output logit)                 → dump output_logit.txt
  → Final prediction

Cũng thực hiện integer-only arithmetic (giống ASIC) cho từng bước.

Ngưỡng phân loại ACTIVE: 0.9 (đồng nhất với Module 1)

Chạy:
  python layer_trace.py --model drowsiness_asic_int8.tflite \\
                        --pixels pixel_dumps/YYYYMMDD_HHMMSS_pixels_hex.txt

Hoặc từ ảnh trực tiếp:
  python layer_trace.py --model drowsiness_asic_int8.tflite --image face.jpg
"""

import numpy as np
import argparse
import os
import sys

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

IMG_SIZE          = 128
SCALE_FACTOR      = 128.0
ACTIVE_THRESHOLD  = 0.9   # Đồng nhất với Module 1 (realtime_cam.py)


# ════════════════════════════════════════════════════════════════════════════
# LOAD INPUT
# ════════════════════════════════════════════════════════════════════════════

def load_from_hex_file(hex_path):
    """Đọc file hex từ Module 2 → int8 array (128,128)"""
    values = []
    with open(hex_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            v = int(line, 16)
            # Unsigned hex → signed int8 (two's complement)
            if v > 127:
                v -= 256
            values.append(v)

    if len(values) != IMG_SIZE * IMG_SIZE:
        print(f"⚠️  Số pixel đọc được ({len(values)}) "
              f"≠ {IMG_SIZE}×{IMG_SIZE}={IMG_SIZE*IMG_SIZE}")

    arr = np.array(values, dtype=np.int8).reshape(IMG_SIZE, IMG_SIZE)
    print(f"  Loaded {len(values)} pixels from {hex_path}")
    print(f"  Shape: {arr.shape}, range: [{arr.min()}, {arr.max()}]")
    return arr


def load_from_image(image_path):
    """Load ảnh → preprocessing → int8 Q0.7"""
    import cv2
    try:
        import mediapipe as mp
        mp_fm = mp.solutions.face_mesh
        mp_dr = mp.solutions.drawing_utils
        dn    = mp_dr._normalized_to_pixel_coordinates
        _l    = set(np.ravel(list(mp_fm.FACEMESH_LEFT_EYE)))
        _r    = set(np.ravel(list(mp_fm.FACEMESH_RIGHT_EYE)))
        EYE   = _l | _r
        MP_OK = True
    except ImportError:
        MP_OK = False
        print("⚠️  MediaPipe không có — bỏ qua landmarks")

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Không đọc được: {image_path}")

    # Detect face
    fc    = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.3, 5)
    if len(faces):
        x, y, w, h = faces[0]
        roi = frame[y:y+h, x:x+w]
        print(f"  Face detected: ({x},{y}) {w}×{h}")
    else:
        roi = frame
        print("  ⚠️  Không detect face — dùng toàn bộ frame")

    # MediaPipe landmarks
    if MP_OK:
        out  = roi.copy()
        conn = mp_dr.DrawingSpec(thickness=1, circle_radius=2, color=(255, 255, 255))
        with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                             refine_landmarks=False,
                             min_detection_confidence=0.5) as fm:
            res = fm.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                for fl in res.multi_face_landmarks:
                    mp_dr.draw_landmarks(image=out, landmark_list=fl,
                                         connections=mp_fm.FACEMESH_TESSELATION,
                                         landmark_drawing_spec=None,
                                         connection_drawing_spec=conn)
                    for idx, lm in enumerate(fl.landmark):
                        if idx in EYE:
                            h_, w_ = roi.shape[:2]
                            pt = dn(lm.x, lm.y, w_, h_)
                            if pt:
                                cv2.circle(out, pt, 3, (255, 255, 255), -1)
        roi = out

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized  = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    q07      = np.clip((resized.astype(np.int16) >> 1), 0, 127).astype(np.int8)
    return q07


# ════════════════════════════════════════════════════════════════════════════
# TFLITE LAYER TRACE
# ════════════════════════════════════════════════════════════════════════════

def get_all_intermediate_tensors(model_path, q07_input):
    """
    Chạy inference và thu thập output của TẤT CẢ intermediate tensors.
    Đây là 'golden reference' — RTL phải match với kết quả này.

    Returns:
        results  : dict[tensor_index → tensor_info]
        inp_int8 : int8 input đã quantize (1,128,128,1)
        out_raw  : int8 output cuối
        logit    : float logit
        prob     : float sigmoid(logit)
        inp_det  : input tensor detail
        out_det  : output tensor detail
    """
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    all_det = interp.get_tensor_details()

    # Quantize input: Q0.7 int8 → TFLite int8 (áp dụng scale/zp của model)
    inp_scale = float(inp_det['quantization_parameters']['scales'][0])
    inp_zp    = int(inp_det['quantization_parameters']['zero_points'][0])

    fp32_in  = (q07_input.astype(np.float32) / SCALE_FACTOR
                )[np.newaxis, :, :, np.newaxis]
    inp_int8 = np.round(fp32_in / inp_scale + inp_zp
                        ).clip(-128, 127).astype(np.int8)

    # Chạy inference
    interp.set_tensor(inp_det['index'], inp_int8)
    interp.invoke()

    # Thu thập tất cả tensor values
    results = {}
    for t in all_det:
        try:
            data = interp.get_tensor(t['index'])
            qp   = t['quantization_parameters']
            results[t['index']] = {
                'name':  t['name'],
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'data':  data.copy(),
                'scale': float(qp['scales'][0])      if len(qp['scales'])      > 0 else None,
                'zp':    int(qp['zero_points'][0])   if len(qp['zero_points']) > 0 else None,
            }
        except Exception:
            pass   # Một số tensor không accessible

    # Kết quả cuối
    out_raw   = interp.get_tensor(out_det['index'])
    out_scale = float(out_det['quantization_parameters']['scales'][0])
    out_zp    = int(out_det['quantization_parameters']['zero_points'][0])
    logit     = float((int(out_raw[0][0]) - out_zp) * out_scale)
    prob      = 1.0 / (1.0 + np.exp(-logit))

    return results, inp_int8, out_raw, logit, prob, inp_det, out_det


# ════════════════════════════════════════════════════════════════════════════
# DUMP LAYER OUTPUTS
# ════════════════════════════════════════════════════════════════════════════

def dump_tensor_to_file(tensor_info, idx, out_dir):
    """
    Dump 1 intermediate tensor → 3 file: .int.txt, .hex.txt, .float.txt
    Format cho RTL: 1 giá trị/dòng, integer.
    """
    data  = tensor_info['data']
    name  = tensor_info['name']
    scale = tensor_info['scale']
    zp    = tensor_info['zp']

    safe = (name.replace('/', '_').replace(';', '_')
               .replace(':', '_').replace('.', '_').strip('_'))
    fname = f"layer_{idx:03d}_{safe}"

    # ── Integer dump ─────────────────────────────────────────────────────
    int_path = os.path.join(out_dir, f"{fname}.int.txt")
    flat     = data.flatten()
    with open(int_path, 'w') as f:
        f.write(f"// Layer output: {name}\n")
        f.write(f"// Index: {idx}\n")
        f.write(f"// Shape: {tensor_info['shape']}\n")
        f.write(f"// Dtype: {tensor_info['dtype']}\n")
        f.write(f"// Scale: {scale}\n")
        f.write(f"// Zero point: {zp}\n")
        f.write(f"// N elements: {len(flat)}\n")
        f.write(f"// Format: integer (raw quantized), 1 per line\n")
        f.write(f"// RTL comparison: compare this file with your sim output\n\n")
        for v in flat:
            f.write(f"{int(v)}\n")

    # ── Hex dump ─────────────────────────────────────────────────────────
    hex_path = os.path.join(out_dir, f"{fname}.hex.txt")
    with open(hex_path, 'w') as f:
        f.write(f"// {name} — hex format\n")
        f.write(f"// Shape: {tensor_info['shape']}, dtype: {tensor_info['dtype']}\n\n")
        for v in flat:
            f.write(f"{int(v) & 0xFF:02X}\n")

    # ── Float dequantized ─────────────────────────────────────────────────
    float_path = os.path.join(out_dir, f"{fname}.float.txt")
    with open(float_path, 'w') as f:
        f.write(f"// {name} — dequantized float32\n")
        f.write(f"// real_value = scale × (int_value - zero_point)\n")
        f.write(f"// scale={scale}, zp={zp}\n\n")
        if scale is not None and zp is not None:
            for v in flat:
                real = float(scale) * (float(v) - float(zp))
                f.write(f"{real:.8f}\n")
        else:
            f.write("// ⚠️  scale hoặc zp không có — không thể dequantize\n")

    return int_path, hex_path, float_path


def dump_input_tensor(inp_int8, out_dir):
    """Dump input int8 tensor (sau khi quantize bởi model)"""
    flat     = inp_int8.flatten()
    int_path = os.path.join(out_dir, "layer_000_INPUT.int.txt")
    hex_path = os.path.join(out_dir, "layer_000_INPUT.hex.txt")

    with open(int_path, 'w') as f:
        f.write("// INPUT TENSOR — Q0.7 int8\n")
        f.write(f"// Shape: {inp_int8.shape}\n")
        f.write(f"// Range: [{int(inp_int8.min())}, {int(inp_int8.max())}]\n\n")
        for v in flat:
            f.write(f"{int(v)}\n")

    with open(hex_path, 'w') as f:
        f.write("// INPUT TENSOR — hex\n")
        f.write(f"// Shape: {inp_int8.shape}\n\n")
        for v in flat:
            f.write(f"{int(v) & 0xFF:02X}\n")

    return int_path, hex_path


# ════════════════════════════════════════════════════════════════════════════
# INTEGER-ONLY ARITHMETIC DEMO
# ════════════════════════════════════════════════════════════════════════════

def _find_first_weight_bias(tensors_all):
    """
    Tìm weight + bias tensor của conv layer đầu tiên theo index nhỏ nhất.
    Không giả định tên tensor — tìm qua dtype và shape.

    Returns: (weight_tensor_info, bias_tensor_info) hoặc (None, None)
    """
    # Tách weight (int8, shape 4D) và bias (int32)
    conv_weights = {
        idx: t for idx, t in tensors_all.items()
        if t['dtype'] == 'int8' and len(t['shape']) == 4
    }
    biases = {
        idx: t for idx, t in tensors_all.items()
        if t['dtype'] == 'int32'
    }

    if not conv_weights:
        return None, None

    # Lấy weight có index nhỏ nhất (= conv layer đầu tiên)
    first_w_idx = min(conv_weights.keys())
    wt = conv_weights[first_w_idx]

    # Tìm bias gần nhất (index nhỏ nhất trong biases)
    bt = biases.get(min(biases.keys())) if biases else None

    return wt, bt


def integer_only_demo(inp_int8, tensors_all, inp_det, out_dir):
    """
    Demo thực hiện MAC operation bằng integer arithmetic thuần túy
    cho 1 output neuron của Conv1, để RTL engineer verify từng bit.
    """
    demo_path = os.path.join(out_dir, "INTEGER_ONLY_DEMO.txt")

    weight_tensor, bias_tensor = _find_first_weight_bias(tensors_all)

    with open(demo_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("INTEGER-ONLY MAC DEMO — Conv1, Output Channel 0, Pixel (0,0)\n")
        f.write("Dùng để verify từng bước trong RTL testbench\n")
        f.write("=" * 70 + "\n\n")

        if weight_tensor is None:
            f.write("// Không tìm được Conv weight tensor 4D\n")
            print("  ⚠️  Không tìm được Conv weight — bỏ qua MAC demo")
            return demo_path

        W_data   = weight_tensor['data']   # TFLite: (Cout, H, W, Cin)
        W_scales = weight_tensor['scale']  # float hoặc array per-channel

        inp_scale = float(inp_det['quantization_parameters']['scales'][0])
        inp_zp    = int(inp_det['quantization_parameters']['zero_points'][0])

        inp_flat = inp_int8[0, :, :, 0]   # (128,128)

        f.write(f"Weight tensor: shape={W_data.shape}, dtype={W_data.dtype}\n")
        f.write(f"  TFLite Conv2D layout: (Cout, H, W, Cin)\n\n")

        f.write("INPUT (top-left 3×3 patch, row 0-2, col 0-2):\n")
        patch = []
        for r in range(3):
            row_vals = []
            for c in range(3):
                v = int(inp_flat[r, c])
                row_vals.append(v)
                patch.append(v)
            f.write(f"  row[{r}]: {row_vals}\n")

        f.write(f"\nINPUT zero_point: {inp_zp}\n")
        f.write(f"(x - zp) values:  {[v - inp_zp for v in patch]}\n")

        # Kernel weights cho output channel 0, input channel 0
        if len(W_data.shape) == 4:
            Co, H, W_k, Ci = W_data.shape
            # TFLite layout: W_data[cout, h, w, cin]
            kernel = W_data[0, :, :, 0].flatten()   # cout=0, cin=0
            f.write(f"\nWEIGHT KERNEL [cout=0, cin=0] (shape {H}×{W_k}):\n")
            for i, v in enumerate(kernel):
                f.write(f"  k[{i // W_k}][{i % W_k}] = {int(v)}\n")
        else:
            f.write("\nWeight shape không phải 4D — bỏ qua\n")
            return demo_path

        f.write(f"\n{'─' * 70}\n")
        f.write(f"MAC COMPUTATION (integer-only):\n")
        f.write(f"  acc = 0\n")
        acc = np.int32(0)
        for i, (x_val, w_val) in enumerate(zip(patch, kernel)):
            x_adj   = np.int32(x_val) - np.int32(inp_zp)
            product = np.int32(w_val) * x_adj
            acc    += product
            f.write(f"  step[{i:2d}]: w={int(w_val):4d}, x={x_val:4d}, "
                    f"(x-zp)={int(x_adj):4d}, prod={int(product):8d}, "
                    f"acc={int(acc):10d}\n")

        if bias_tensor is not None:
            b_val = int(bias_tensor['data'].flatten()[0])
            acc  += np.int32(b_val)
            f.write(f"  + bias[0] = {b_val}: acc = {int(acc)}\n")

        # Scale cho output channel 0
        if isinstance(W_scales, np.ndarray):
            sw = float(W_scales[0])
        else:
            sw = float(W_scales)

        # Lấy scale của activation output (nếu có); dùng inp_scale làm approx
        # RTL engineer: thay bằng scale của intermediate tensor thực tế
        out_scale_approx = inp_scale

        M       = inp_scale * sw / out_scale_approx
        shift   = 0
        M_norm  = M
        while M_norm < 0.5 and shift < 31:
            M_norm *= 2
            shift  += 1
        M_int32     = int(round(M_norm * (2**30)))
        total_shift = shift + 30

        result_i64     = np.int64(acc) * np.int64(M_int32)
        result_i32     = int(result_i64) >> total_shift
        result_clipped = max(-128, min(127, result_i32))

        f.write(f"\nREQUANTIZATION:\n")
        f.write(f"  M = S_in × S_w / S_out = "
                f"{inp_scale:.8f} × {sw:.8f} / {out_scale_approx:.8f}\n")
        f.write(f"    = {M:.10f}\n")
        f.write(f"  M_int32    = {M_int32}  (Q1.30)\n")
        f.write(f"  total_shift= {total_shift}\n")
        f.write(f"  int64 = acc({int(acc)}) × M_int32({M_int32}) = {int(result_i64)}\n")
        f.write(f"  >> {total_shift} = {result_i32}\n")
        f.write(f"  clip(-128,127) = {result_clipped}\n")
        f.write(f"\n  → Conv1 output[0,0,0,ch=0] (before ReLU): {result_clipped}\n")
        f.write(f"  → After ReLU: {max(0, result_clipped)}\n")

        f.write(f"\n{'=' * 70}\n")
        f.write("Verilog equivalent:\n")
        f.write(f"  wire signed [31:0] acc;            // Sau MAC + bias\n")
        f.write(f"  wire signed [63:0] scaled;         // acc × M_int32\n")
        f.write(f"  wire signed [31:0] shifted;        // scaled >> {total_shift}\n\n")
        f.write(f"  assign scaled  = $signed(acc) * $signed(32'd{M_int32});\n")
        f.write(f"  assign shifted = scaled >>> {total_shift};\n")
        f.write(f"  assign q_out   = (shifted > 127)  ? 8'sd127  :\n")
        f.write(f"                   (shifted < -128) ? -8'sd128 :\n")
        f.write(f"                   shifted[7:0];\n")
        f.write(f"  // ReLU: assign relu_out = q_out[7] ? 8'd0 : q_out;\n")

    print(f"  💾 {demo_path}  ← step-by-step MAC verify cho RTL")
    return demo_path


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Layer-by-layer inference trace để so sánh với RTL testbench")
    parser.add_argument("--model",  default="drowsiness_asic_int8.tflite")
    parser.add_argument("--pixels", default=None,
                        help="File hex từ Module 2 (pixels_hex.txt)")
    parser.add_argument("--image",  default=None,
                        help="Dùng ảnh trực tiếp thay vì pixel file")
    parser.add_argument("--out",    default="layer_traces",
                        help="Thư mục output")
    args = parser.parse_args()

    if not args.pixels and not args.image:
        print("❌ Cần chỉ định --pixels hoặc --image")
        sys.exit(1)

    print("=" * 70)
    print("MODULE 4 — LAYER-BY-LAYER TRACE CHO RTL COMPARISON")
    print("=" * 70)

    os.makedirs(args.out, exist_ok=True)

    # Load input
    print("\n📂 Loading input...")
    if args.pixels:
        q07 = load_from_hex_file(args.pixels)
    else:
        q07 = load_from_image(args.image)

    print(f"  Q0.7 int8 input: shape={q07.shape}, "
          f"range=[{q07.min()}, {q07.max()}]")

    # Run inference + collect all intermediate tensors
    print(f"\n⚙️  Running inference + collecting layer outputs...")
    tensors_all, inp_int8, out_raw, logit, prob, inp_det, out_det = \
        get_all_intermediate_tensors(args.model, q07)

    # Final prediction — ngưỡng đồng nhất với Module 1
    label = "ACTIVE 😊" if prob >= ACTIVE_THRESHOLD else "FATIGUE 😴"
    conf  = prob * 100 if prob >= ACTIVE_THRESHOLD else (1 - prob) * 100
    print(f"\n{'=' * 60}")
    print(f"🎯 PREDICTION  : {label}  ({conf:.1f}% confidence)")
    print(f"   logit        = {logit:.4f}")
    print(f"   prob_active  = {prob:.4f}")
    print(f"   threshold    = {ACTIVE_THRESHOLD:.0%}")
    print(f"{'=' * 60}")

    # Dump input
    print(f"\n💾 Dumping tensors to {args.out}/...")
    ip, ih = dump_input_tensor(inp_int8, args.out)
    print(f"  [000] INPUT → {os.path.basename(ip)}")

    # Dump all intermediate tensors
    dump_count = 0
    for tidx, tinfo in sorted(tensors_all.items()):
        data = tinfo['data']
        if data.size == 0:
            continue
        if data.size > 1_000_000:
            print(f"  [{tidx:3d}] SKIP (too large: {data.size:,}): {tinfo['name']}")
            continue

        ip, ih, ifp = dump_tensor_to_file(tinfo, tidx, args.out)
        print(f"  [{tidx:3d}] {tinfo['name'][:50]:<50}  "
              f"shape={str(tinfo['shape']):<18}  n={data.size:6,}  → dumped")
        dump_count += 1

    # Integer-only demo
    print(f"\n📐 Generating integer-only MAC demo...")
    integer_only_demo(inp_int8, tensors_all, inp_det, args.out)

    # RTL comparison guide
    guide_path = os.path.join(args.out, "RTL_COMPARISON_GUIDE.txt")
    with open(guide_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HƯỚNG DẪN SO SÁNH VỚI RTL TESTBENCH\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. FLOW TỔNG QUÁT:\n")
        f.write("   Python (TFLite golden) → dump files → diff với RTL sim output\n\n")
        f.write("2. INPUT:\n")
        f.write(f"   File: layer_000_INPUT.hex.txt\n")
        f.write(f"   $readmemh(\"layer_000_INPUT.hex.txt\", pixel_mem);\n")
        f.write(f"   Format: 1 giá trị/dòng, 2-digit hex, two's complement int8\n\n")
        f.write("3. SO SÁNH TỪNG LAYER:\n")
        f.write("   Sau mỗi layer trong RTL sim, dump output → so sánh với *.int.txt\n")
        f.write("   Tolerance: 0 LSB (exact) hoặc ±1 LSB do rounding\n\n")
        f.write("4. PREDICTION CUỐI:\n")
        f.write(f"   logit     = {logit:.6f}\n")
        f.write(f"   prob      = {prob:.6f}  (sigmoid(logit))\n")
        f.write(f"   threshold = {ACTIVE_THRESHOLD:.0%}  (đồng nhất với Module 1)\n")
        f.write(f"   label     = {label}\n")
        f.write(f"   out_raw   = {int(out_raw[0][0])}  (int8 quantized)\n")
        out_s  = float(out_det['quantization_parameters']['scales'][0])
        out_zp = int(out_det['quantization_parameters']['zero_points'][0])
        f.write(f"   out_scale = {out_s:.8f}\n")
        f.write(f"   out_zp    = {out_zp}\n\n")
        f.write("5. ĐIỀU KIỆN PASS:\n")
        f.write("   - out_raw (int8) match chính xác\n")
        f.write("   - Mỗi intermediate layer match ±1 LSB\n")
        f.write("   - Label (ACTIVE/FATIGUE) match\n\n")
        f.write("6. FILES TRONG THƯ MỤC NÀY:\n")
        f.write("   *.int.txt   — integer values (decimal) để so sánh\n")
        f.write("   *.hex.txt   — hex values cho $readmemh\n")
        f.write("   *.float.txt — dequantized float để debug semantics\n")
        f.write("   INTEGER_ONLY_DEMO.txt — step-by-step MAC verify\n")

    print(f"  💾 {guide_path}")

    print(f"\n{'=' * 70}")
    print(f"✅ TRACE HOÀN THÀNH: {dump_count} tensors dumped")
    print(f"{'=' * 70}")
    print(f"📁 Output: {args.out}/")
    print(f"\n🎯 Prediction: {label}  ({conf:.1f}%)")
    print(f"\n📌 Workflow cho RTL verification:")
    print(f"  1. Load layer_000_INPUT.hex.txt vào RTL testbench")
    print(f"  2. Chạy simulation")
    print(f"  3. So sánh sim output với các file *.int.txt")
    print(f"  4. Đọc RTL_COMPARISON_GUIDE.txt để biết tolerance")
    print(f"\n💡 Nếu mismatch xảy ra:")
    print(f"  • Kiểm tra INTEGER_ONLY_DEMO.txt cho Conv1")
    print(f"  • Verify requant params từ Module 3/REQUANT_PARAMS.txt")
    print(f"  • Đảm bảo bias là int32, không phải int8")
    print(f"  • TFLite Conv2D weight layout: (Cout, H, W, Cin)")


if __name__ == "__main__":
    main()