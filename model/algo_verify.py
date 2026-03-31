"""
MODULE 5 — ALGORITHM VERIFICATION REFERENCE GENERATOR
=======================================================
Sinh ra file mô tả thuật toán từng layer của model drowsiness_asic_int8.tflite
với đủ detail để phần cứng (RTL/ASIC) có thể so sánh (verify) từng bước.

KIẾN TRÚC MODEL (drowsiness_asic — build_asic_model):
  Input: (128, 128, 1)  Q0.7 int8
  Block 1: SeparableConv2D(32,  3×3, padding=same, use_bias=False) + ReLU → MaxPool(2×2)
  Block 2: SeparableConv2D(64,  3×3, padding=same, use_bias=False) + ReLU → MaxPool(2×2)
  Block 3: SeparableConv2D(128, 3×3, padding=same, use_bias=False) + ReLU → MaxPool(2×2)
  Block 4: SeparableConv2D(128, 3×3, padding=same, use_bias=False) + ReLU (no pool)
  GAP → Dense(32, use_bias=True) + ReLU → Dense(1, use_bias=True)

  LƯU Ý: Tất cả SeparableConv2D (DW + PW) đều dùng use_bias=False.
          Chỉ FC1 (Dense 32) và FC2 (Dense 1) mới có bias.
          Trong TFLite int8, bias vẫn được fold vào từ BatchNorm nếu có;
          nếu không có BN thì bias conv = 0 hoặc không tồn tại trong tensor.

TENSOR MAPPING THỰC TẾ (từ debug_tensors.py):
  Do XNNPACK delegate chỉ expose input/output/weights — intermediate activations
  (conv/pool outputs) KHÔNG accessible. Script dùng TFLite reference interpreter
  (không có XNNPACK) để lấy đầy đủ intermediate tensors.

  Weights accessible sau invoke:
    idx=2  → DW Block1  [1,3,3,1]      (bị misclassify là activation vì n_scales=1)
    idx=5  → PW Block1  [32,1,1,1]
    idx=9  → DW Block2  [1,3,3,32]
    idx=12 → PW Block2  [64,1,1,32]
    idx=16 → DW Block3  [1,3,3,64]
    idx=19 → PW Block3  [128,1,1,64]
    idx=23 → DW Block4  [1,3,3,128]
    idx=26 → PW Block4  [128,1,1,128]
    idx=30 → FC1 weight [32,128]
    idx=33 → FC2 weight [1,32]         (bị misclassify là activation vì n_scales=1)

  Biases (Conv layers dùng use_bias=False → bias tensor = zeros hoặc không có):
    idx=3  → DW Block1 bias  [1]   ← zero (use_bias=False)
    idx=6  → PW Block1 bias  [32]  ← zero (use_bias=False)
    idx=10 → DW Block2 bias  [32]  ← zero (use_bias=False)
    idx=13 → PW Block2 bias  [64]  ← zero (use_bias=False)
    idx=17 → DW Block3 bias  [64]  ← zero (use_bias=False)
    idx=20 → PW Block3 bias  [128] ← zero (use_bias=False)
    idx=24 → DW Block4 bias  [128] ← zero (use_bias=False)
    idx=27 → PW Block4 bias  [128] ← zero (use_bias=False)
    idx=31 → FC1 bias  [32]        ← có giá trị thực (use_bias=True)
    idx=34 → FC2 bias  [1]         ← có giá trị thực (use_bias=True)

  Accessible intermediate activations (XNNPACK):
    idx=29 → GAP output    [1,128]
    idx=32 → FC1+ReLU out  [1,32]
    idx=35 → Final output  [1,1]

  NOTE: Conv/Pool layer outputs KHÔNG accessible → file input/output của các
  layer đó sẽ được tính lại bằng numpy simulation (float32 approximation).
  Các file golden chính xác nhất là: GAP, FC1, FC2, OUTPUT.

Mỗi layer → 1 thư mục riêng chứa:
  ALGO.txt         — Thuật toán integer-only, công thức, requant params
  input.hex.txt    — Input int8 hex ($readmemh)
  input.int.txt    — Input int8 decimal
  output.hex.txt   — Output int8 hex ($readmemh)  [golden reference]
  output.int.txt   — Output int8 decimal          [golden reference]
  output.float.txt — Output dequantized float
  kernel.hex.txt   — Weight int8 hex (Conv/Dense)
  bias.int.txt     — Bias int32 (nếu có)
  params.json      — Hyperparameters machine-readable

Chạy:
  python algo_verify.py --model drowsiness_asic_int8.tflite \\
                        --pixels pixel_dumps/YYYYMMDD_pixels_hex.txt \\
                        --out    verify_output/

  python algo_verify.py --model drowsiness_asic_int8.tflite \\
                        --image  face.jpg --out verify_output/
"""

import numpy as np
import argparse
import os
import sys
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (đồng nhất với các module khác)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE         = 128
SCALE_FACTOR     = 128.0      # Q0.7: int8 / 128 = float in [-1, 1)
ACTIVE_THRESHOLD = 0.9        # Đồng nhất với Module 1 (realtime_cam.py)

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: INPUT LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_from_hex_file(hex_path):
    """Đọc file hex từ Module 2 (capture_pixel_dump.py) → int8 (128,128)"""
    values = []
    with open(hex_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            v = int(line, 16)
            if v > 127:
                v -= 256
            values.append(v)

    if len(values) != IMG_SIZE * IMG_SIZE:
        print(f"  ⚠️  Pixel count ({len(values)}) ≠ {IMG_SIZE}²={IMG_SIZE*IMG_SIZE}")

    arr = np.array(values, dtype=np.int8).reshape(IMG_SIZE, IMG_SIZE)
    print(f"  Loaded {len(values)} pixels từ {hex_path}")
    print(f"  Shape={arr.shape}, range=[{arr.min()}, {arr.max()}]")
    return arr


def load_from_image(image_path):
    """Load ảnh → grayscale → resize → Q0.7 int8"""
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Không đọc được: {image_path}")

    fc   = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.3, 5)
    if len(faces):
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        print(f"  Face detected: ({x},{y}) {w}×{h}")
    else:
        roi = gray
        print("  ⚠️  No face — dùng toàn frame")

    resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    q07     = np.clip((resized.astype(np.int16) >> 1), 0, 127).astype(np.int8)
    return q07


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: TFLITE INFERENCE — COLLECT ALL TENSORS + OPS
# ═════════════════════════════════════════════════════════════════════════════

def run_inference_collect(model_path, q07_input):
    """
    Quantize input → invoke → thu thập TẤT CẢ tensor values sau inference.

    Returns:
        interp      : Interpreter (đã invoke)
        inp_det     : input tensor detail
        out_det     : output tensor detail
        all_details : danh sách tất cả tensor details
        inp_int8    : quantized input (1,128,128,1) int8
        logit       : float logit (dequantized output)
        prob        : sigmoid(logit)
    """
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_det     = interp.get_input_details()[0]
    out_det     = interp.get_output_details()[0]
    all_details = interp.get_tensor_details()

    inp_scale = float(inp_det['quantization_parameters']['scales'][0])
    inp_zp    = int(inp_det['quantization_parameters']['zero_points'][0])

    # Dequant: int8 → float32, rồi requant vào TFLite int8
    fp32_in  = q07_input.astype(np.float32) / SCALE_FACTOR
    fp32_in  = fp32_in[np.newaxis, :, :, np.newaxis]
    inp_int8 = np.round(fp32_in / inp_scale + inp_zp).clip(-128, 127).astype(np.int8)

    interp.set_tensor(inp_det['index'], inp_int8)
    interp.invoke()

    # Lấy kết quả cuối
    out_int8  = interp.get_tensor(out_det['index'])
    out_scale = float(out_det['quantization_parameters']['scales'][0])
    out_zp    = int(out_det['quantization_parameters']['zero_points'][0])
    logit     = (float(out_int8.flatten()[0]) - out_zp) * out_scale
    prob      = 1.0 / (1.0 + np.exp(-logit))

    return interp, inp_det, out_det, all_details, inp_int8, logit, prob





# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: FILE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def safe_name(s):
    for c in r'/\;:. ':
        s = s.replace(c, '_')
    return s.strip('_')


def write_hex_file(path, data_int8, header=""):
    """Ghi int8 array → hex file (2-digit, $readmemh compatible)"""
    flat = np.array(data_int8).flatten()
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + "\n")
        f.write(f"// $readmemh(\"{os.path.basename(path)}\", mem);\n")
        f.write(f"// {len(flat)} values, int8 two's complement\n\n")
        for v in flat:
            f.write(f"{int(v) & 0xFF:02X}\n")


def write_int_file(path, data, header=""):
    """Ghi array → decimal text (1 value/dòng)"""
    flat = np.array(data).flatten()
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + "\n")
        f.write(f"// {len(flat)} values, decimal\n\n")
        for v in flat:
            f.write(f"{int(v)}\n")


def write_float_file(path, data_int8, scale, zp, header=""):
    """Ghi dequantized float (debug semantics)"""
    flat   = np.array(data_int8, dtype=np.float32).flatten()
    floats = (flat - zp) * scale
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + "\n")
        f.write(f"// Dequantized: float = (int8 - {zp}) × {scale:.10f}\n")
        f.write(f"// {len(floats)} values\n\n")
        for v in floats:
            f.write(f"{float(v):.8f}\n")


def dump_weight_files(layer_dir, prefix, tensor_info, layout_note=""):
    """
    Dump weight tensor ra đủ 4 format giống weight_dump.py:
      <prefix>.dec.txt    — decimal int8 (1 value/dòng)
      <prefix>.hex.txt    — hex int8 ($readmemh, 2-digit)
      <prefix>.scales.txt — per-channel scales + Q8.24 fixed-point
      <prefix>.shaped.txt — shaped view (chỉ với tensor nhỏ ≤ 2048 elements)

    Args:
        layer_dir   : thư mục output của layer
        prefix      : tên prefix file (ví dụ "kernel_dw", "kernel_pw", "weight_fc")
        tensor_info : dict với keys 'data'(int8), 'shape', 'scales_arr', 'zps_arr', 'name'
        layout_note : ghi chú layout cho header (ví dụ "[1, KH, KW, Cout]")
    """
    data       = np.array(tensor_info['data'], dtype=np.int8)
    shape      = list(data.shape)
    flat       = data.flatten()
    scales     = tensor_info['scales_arr']
    zps        = tensor_info['zps_arr'] if tensor_info.get('zps_arr') is not None \
                 else np.zeros(len(scales), dtype=np.int32)
    name       = tensor_info.get('name', prefix)
    n_elements = int(data.size)
    layout     = layout_note or str(shape)

    header_common = (
        f"// Tensor : {name}\n"
        f"// Shape  : {shape}  {layout}\n"
        f"// Dtype  : int8  (quantized weight, zero_point=0 per-channel)\n"
        f"// N_ch_scales: {len(scales)}\n"
    )

    # ── 1. Decimal (.dec.txt) ────────────────────────────────────────────────
    p = os.path.join(layer_dir, f"{prefix}.dec.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(header_common)
        f.write(f"// {n_elements} values, signed decimal int8\n\n")
        for v in flat:
            f.write(f"{int(v)}\n")

    # ── 2. Hex ($readmemh) (.hex.txt) ────────────────────────────────────────
    p = os.path.join(layer_dir, f"{prefix}.hex.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(header_common)
        f.write(f"// $readmemh(\"{prefix}.hex.txt\", weight_mem);\n")
        f.write(f"// {n_elements} values, 2-digit hex, two's complement\n\n")
        for v in flat:
            f.write(f"{int(v) & 0xFF:02X}\n")

    # ── 3. Scales per channel (.scales.txt) ──────────────────────────────────
    p = os.path.join(layer_dir, f"{prefix}.scales.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(header_common)
        f.write(f"// real_value = scale[ch] × int8_value  (zero_point=0 for weight)\n\n")
        f.write(f"// ch,  scale_f32,             scale_Q8.24,    zp\n")
        for i, (s, z) in enumerate(zip(scales,
                                        zps if len(zps) == len(scales)
                                        else np.zeros(len(scales), dtype=np.int32))):
            q24 = int(round(float(s) * 2**24)) & 0xFFFFFFFF
            f.write(f"{i:5d},  {float(s):.12f},  {q24:08X},  zp={int(z)}\n")

    # ── 4. Shaped view (.shaped.txt) — chỉ khi tensor nhỏ ───────────────────
    if n_elements <= 4096 and len(shape) in (2, 4):
        p = os.path.join(layer_dir, f"{prefix}.shaped.txt")
        with open(p, 'w', encoding='utf-8') as f:
            f.write(header_common)
            f.write(f"// Shaped view\n\n")
            if len(shape) == 4:
                if shape[0] == 1:
                    # DW layout: [1, KH, KW, Cout]
                    _, KH, KW, Cout = shape
                    f.write(f"// DW layout: [1, KH={KH}, KW={KW}, Cout={Cout}]\n\n")
                    for co in range(Cout):
                        sw = float(scales[co % len(scales)])
                        f.write(f"// ch={co:3d}  scale={sw:.8f}\n")
                        for kh in range(KH):
                            row = ' '.join(f"{int(data[0, kh, kw, co]):4d}"
                                           for kw in range(KW))
                            f.write(f"//   kh={kh}: {row}\n")
                        f.write("\n")
                else:
                    # PW layout: [Cout, 1, 1, Cin]
                    Cout, _, _, Cin = shape
                    f.write(f"// PW layout: [Cout={Cout}, 1, 1, Cin={Cin}]\n\n")
                    for co in range(Cout):
                        sw  = float(scales[co % len(scales)])
                        row = ' '.join(f"{int(data[co, 0, 0, ci]):4d}"
                                       for ci in range(Cin))
                        f.write(f"// co={co:3d}  scale={sw:.8f}:  {row}\n")
            else:
                # Dense: [Cout, Cin]
                Cout, Cin = shape
                f.write(f"// Dense layout: [Cout={Cout}, Cin={Cin}]\n\n")
                for co in range(Cout):
                    sw  = float(scales[co % len(scales)])
                    row = ' '.join(f"{int(data[co, ci]):4d}" for ci in range(Cin))
                    f.write(f"// out[{co:3d}]  scale={sw:.8f}:  {row}\n")


def write_bias_int(path, bias_int32, header=""):
    """Ghi bias int32 → decimal + hex"""
    flat = np.array(bias_int32).flatten()
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + "\n")
        f.write(f"// {len(flat)} bias values, int32\n")
        f.write(f"// idx,  decimal,          hex_32bit\n\n")
        for i, v in enumerate(flat):
            iv = int(v)
            f.write(f"{i:5d},  {iv:14d},  {iv & 0xFFFFFFFF:08X}\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: ALGORITHM DESCRIPTION WRITERS
# ═════════════════════════════════════════════════════════════════════════════

def write_algo_input(out_dir, inp_int8, inp_det):
    """Viết mô tả và dump tensor INPUT (sau quantize từ Q0.7)"""
    layer_dir = os.path.join(out_dir, "layer_00_INPUT")
    os.makedirs(layer_dir, exist_ok=True)

    scale = float(inp_det['quantization_parameters']['scales'][0])
    zp    = int(inp_det['quantization_parameters']['zero_points'][0])
    data  = inp_int8.copy()

    # ALGO.txt
    with open(os.path.join(layer_dir, "ALGO.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write("LAYER: INPUT — Quantization từ Q0.7 pixel → TFLite int8\n")
        f.write("=" * 72 + "\n\n")
        f.write("MÔ TẢ:\n")
        f.write("  Pixel gốc là Q0.7 int8 (range [-128, 127], scale=1/128)\n")
        f.write("  TFLite model sử dụng scale/zero_point riêng của nó.\n")
        f.write("  Bước này convert từ Q0.7 sang TFLite quantization.\n\n")
        f.write("THUẬT TOÁN (integer-friendly):\n")
        f.write("  1. Dequant từ Q0.7:\n")
        f.write("       float_val = pixel_q07 / 128.0\n")
        f.write(f"  2. Requant sang TFLite int8 (scale={scale:.10f}, zp={zp}):\n")
        f.write(f"       inp_int8 = round(float_val / {scale:.10f} + {zp})\n")
        f.write(f"       inp_int8 = clip(inp_int8, -128, 127)\n\n")
        f.write("COMBINED (tránh dùng float nếu có thể):\n")
        f.write(f"  ratio = (1/128) / scale = 1/128 / {scale:.10f}\n")
        ratio = (1.0/128.0) / scale
        # Xấp xỉ bằng fixed-point Q8.24
        ratio_q24 = int(round(ratio * 2**24))
        f.write(f"         = {ratio:.8f}  ≈  {ratio_q24} / 2^24  (Q8.24 approx)\n")
        f.write(f"  inp_int8 = clip(round(pixel_q07 × {ratio_q24} >> 24) + {zp}, -128, 127)\n\n")
        f.write("INPUT SPECS:\n")
        f.write(f"  Shape   : {list(data.shape)}\n")
        f.write(f"  Dtype   : int8\n")
        f.write(f"  Scale   : {scale:.10f}\n")
        f.write(f"  ZP      : {zp}\n")
        f.write(f"  Range   : [{int(data.min())}, {int(data.max())}]\n\n")
        f.write("OUTPUT FILES:\n")
        f.write("  input.hex.txt    — $readmemh format\n")
        f.write("  input.int.txt    — decimal\n")
        f.write("  input.float.txt  — dequantized float (debug)\n")

    hdr = f"// INPUT tensor: shape={list(data.shape)}, scale={scale:.10f}, zp={zp}"
    write_hex_file(os.path.join(layer_dir, "input.hex.txt"),   data, hdr)
    write_int_file(os.path.join(layer_dir, "input.int.txt"),   data, hdr)
    write_float_file(os.path.join(layer_dir, "input.float.txt"), data, scale, zp, hdr)

    print(f"  [00] INPUT                     shape={list(data.shape)}")
    return layer_dir


def write_algo_depthwise_conv(out_dir, layer_num, layer_name,
                              in_tensor, dw_kernel_tensor, bias_tensor,
                              out_tensor, stride, padding_mode):
    """
    Viết ALGO.txt và dump files cho một DepthwiseConv2D layer.
    TFLite SeparableConv2D = DepthwiseConv2D + PointwiseConv2D (1×1 Conv2D).
    """
    layer_dir = os.path.join(out_dir, f"layer_{layer_num:02d}_{safe_name(layer_name)}_DW")
    os.makedirs(layer_dir, exist_ok=True)

    in_s   = in_tensor['scale']
    in_zp  = in_tensor['zp']
    in_d   = in_tensor['data']

    dw_d      = dw_kernel_tensor['data']    # shape: (1, KH, KW, Cout)
    dw_scales = dw_kernel_tensor['scales_arr']

    bias_d    = bias_tensor['data'] if bias_tensor else None
    bias_s    = bias_tensor['scale'] if bias_tensor else None

    out_d  = out_tensor['data']
    out_s  = out_tensor['scale']
    out_zp = out_tensor['zp']

    KH, KW = dw_d.shape[1], dw_d.shape[2]
    Cout   = dw_d.shape[3]
    N, IH, IW, Cin = in_d.shape

    with open(os.path.join(layer_dir, "ALGO.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write(f"LAYER {layer_num:02d}: DEPTHWISE CONV2D — {layer_name}\n")
        f.write("=" * 72 + "\n\n")
        f.write("MÔ TẢ:\n")
        f.write("  DepthwiseConv2D áp dụng 1 filter riêng cho mỗi input channel.\n")
        f.write("  Không trộn channels (ngược với Conv2D thông thường).\n")
        f.write("  Là bước đầu tiên của SeparableConv2D.\n\n")
        f.write("SPECS:\n")
        f.write(f"  Input  : shape={list(in_d.shape)}, scale={in_s:.10f}, zp={in_zp}\n")
        f.write(f"  Kernel : shape={list(dw_d.shape)}  [1, KH, KW, Cout]\n")
        f.write(f"           KH={KH}, KW={KW}\n")
        f.write(f"  Stride : {stride}\n")
        f.write(f"  Padding: {padding_mode}  (SAME → pad để output H/W = ceil(IH/stride))\n")
        f.write(f"  Output : shape={list(out_d.shape)}, scale={out_s:.10f}, zp={out_zp}\n\n")

        if padding_mode.upper() == 'SAME':
            pad_h = max((KH - 1), 0) // 2
            pad_w = max((KW - 1), 0) // 2
        else:
            pad_h = pad_w = 0
        f.write(f"PADDING (SAME): pad_top=pad_bottom={pad_h}, pad_left=pad_right={pad_w}\n\n")

        # ── Giải thích biến ──────────────────────────────────────────────────
        f.write("GIẢI THÍCH CÁC BIẾN:\n")
        f.write("  ┌─────────────┬──────────┬────────────────────────────────────────────┐\n")
        f.write("  │ Biến        │ Kiểu     │ Ý nghĩa                                    │\n")
        f.write("  ├─────────────┼──────────┼────────────────────────────────────────────┤\n")
        f.write("  │ n           │ int      │ batch index (luôn = 0 với ASIC 1 ảnh)      │\n")
        f.write(f"  │ IH, IW      │ int      │ input height={IH}, width={IW}             │\n")
        f.write(f"  │ Cin         │ int      │ input channels = {Cin}                    │\n")
        f.write(f"  │ Cout        │ int      │ output channels = {Cout}                  │\n")
        f.write(f"  │ KH, KW      │ int      │ kernel height={KH}, width={KW}            │\n")
        f.write(f"  │ stride      │ int      │ bước nhảy spatial = {stride}              │\n")
        f.write(f"  │ pad_top/left│ int      │ số pixel padding = {pad_h}                │\n")
        f.write("  │ oh, ow      │ int      │ output row, col đang tính                  │\n")
        f.write("  │ kh, kw      │ int      │ vị trí trong cửa sổ kernel 3×3             │\n")
        f.write("  │ ih, iw      │ int      │ vị trí tương ứng trên input (sau map)      │\n")
        f.write("  │ ch          │ int      │ channel index (DW: mỗi ch xử lý độc lập)   │\n")
        f.write("  │ x_int8      │ int8     │ 1 pixel input tại [ih, iw, ch]             │\n")
        f.write("  │ w_int8      │ int8     │ 1 weight tại kernel[0, kh, kw, ch]         │\n")
        f.write("  │ x_zp        │ int8     │ zero-point của input (cộng vào lúc quant)  │\n")
        f.write("  │ acc         │ int32    │ accumulator MAC — tích luỹ KH×KW tích       │\n")
        f.write("  │ bias[ch]    │ int32    │ hệ số bias đã fold BN, 1 giá trị/channel   │\n")
        f.write("  │ S_in        │ float32  │ scale của input tensor                      │\n")
        f.write("  │ S_weight[ch]│ float32  │ scale của weight channel ch (per-channel)   │\n")
        f.write("  │ S_out       │ float32  │ scale của output tensor                     │\n")
        f.write("  │ M_int32     │ int32    │ multiplier requant (Q1.30 fixed-point)      │\n")
        f.write("  │ total_shift │ int      │ số bit shift phải sau nhân M_int32          │\n")
        f.write("  │ out_zp      │ int8     │ zero-point output (cộng sau shift)          │\n")
        f.write("  │ out_val     │ int8     │ output cuối cùng sau clip [-128, 127]       │\n")
        f.write("  └─────────────┴──────────┴────────────────────────────────────────────┘\n\n")

        f.write("VỊ TRÍ TRONG BỘ NHỚ / FILE:\n")
        f.write(f"  INPUT  [n=0, ih, iw, ch]  →  offset = ih×{IW*Cin} + iw×{Cin} + ch\n")
        OH_str = f"ceil({IH}/{stride})" if stride > 1 else str(IH)
        OW_str = f"ceil({IW}/{stride})" if stride > 1 else str(IW)
        f.write(f"  OUTPUT [n=0, oh, ow, ch]  →  offset = oh×{int(np.ceil(IH/stride))*Cout} + ow×{Cout} + ch\n")
        f.write(f"  KERNEL [0,  kh, kw, ch]   →  offset = kh×{KW*Cout} + kw×{Cout} + ch\n")
        f.write(f"  BIAS   [ch]               →  offset = ch  (1 int32 mỗi output channel)\n\n")
        f.write("  Lưu ý: tất cả tensor lưu theo thứ tự NHWC (row-major cuối cùng là channel)\n")
        f.write("         File hex/int dump theo thứ tự flatten() = duyệt từ chiều ngoài vào trong\n\n")

        f.write("MINH HOẠ KERNEL 3×3 (ch=0) trong file kernel_dw.shaped.txt:\n")
        f.write(f"  kernel[0, kh=0, kw=0, ch=0]  kernel[0, kh=0, kw=1, ch=0]  kernel[0, kh=0, kw=2, ch=0]\n")
        f.write(f"  kernel[0, kh=1, kw=0, ch=0]  kernel[0, kh=1, kw=1, ch=0]  kernel[0, kh=1, kw=2, ch=0]\n")
        f.write(f"  kernel[0, kh=2, kw=0, ch=0]  kernel[0, kh=2, kw=1, ch=0]  kernel[0, kh=2, kw=2, ch=0]\n")
        f.write(f"  └── đây là filter áp lên channel 0 của input, quét theo hàng trước\n\n")

        f.write("THUẬT TOÁN INTEGER-ONLY:\n")
        f.write("  // Với mỗi output element [n, oh, ow, ch]:\n")
        f.write("  int32 acc = 0\n")
        f.write("  for kh in 0..KH-1:\n")
        f.write("    for kw in 0..KW-1:\n")
        f.write("      ih = oh × stride + kh - pad_top\n")
        f.write("      iw = ow × stride + kw - pad_left\n")
        f.write("      if 0 ≤ ih < IH and 0 ≤ iw < IW:\n")
        f.write("        x_int8 = input[n, ih, iw, ch]           // int8\n")
        f.write("        w_int8 = kernel[0, kh, kw, ch]          // int8\n")
        f.write("        acc   += (int32)(x_int8 - x_zp) × (int32)(w_int8)\n")
        f.write("        // NOTE: weight zero_point = 0 (per-channel quant)\n")
        if bias_d is not None:
            f.write("  acc += bias[ch]                               // int32\n")
        f.write("\n")
        f.write("  // Requantization:\n")
        f.write("  M = S_in × S_weight[ch] / S_out\n")
        f.write("  Biểu diễn M = M_int32 / 2^total_shift  (Q1.30 format)\n")
        f.write("  int64 scaled  = (int64)acc × (int64)M_int32\n")
        f.write("  int32 shifted = scaled >>> total_shift\n")
        f.write("  int8  out_val = clip(shifted + out_zp, -128, 127)\n\n")
        f.write("LƯU Ý BIAS:\n")
        f.write("  Layer này dùng use_bias=False → bias tensor = zeros.\n")
        f.write("  Bước cộng bias (acc += bias[ch]) không đóng góp gì.\n")
        f.write("  RTL có thể bỏ qua bước cộng bias cho DW/PW layers.\n\n")

        f.write("REQUANTIZATION PARAMS (per output channel):\n")
        f.write(f"  S_in  = {in_s:.12f}\n")
        f.write(f"  S_out = {out_s:.12f}\n")
        f.write(f"  {'ch':>4}  {'S_w':>14}  {'M_float':>18}  {'M_int32':>12}  {'shift':>6}\n")
        f.write(f"  {'─'*62}\n")
        for ch in range(min(Cout, len(dw_scales))):
            sw = float(dw_scales[ch])
            M  = in_s * sw / out_s
            shift, M_norm = 0, M
            while M_norm < 0.5 and shift < 31:
                M_norm *= 2; shift += 1
            M_i32 = int(round(M_norm * 2**30))
            f.write(f"  {ch:>4}  {sw:>14.10f}  {M:>18.12f}  {M_i32:>12d}  {shift+30:>6d}\n")

        f.write(f"\nVerilog DATAPATH TEMPLATE:\n")
        f.write(f"  // Sau khi tính acc (int32):\n")
        f.write(f"  wire signed [63:0] scaled  = $signed(acc) * $signed(M_int32_ch);\n")
        f.write(f"  wire signed [31:0] shifted = scaled >>> TOTAL_SHIFT_ch;\n")
        f.write(f"  wire signed  [7:0] out_val = clip(shifted + {out_zp}, -128, 127);\n")
        f.write(f"  // clip: dùng saturation logic, không phải modulo\n")

    # Dump files
    hdr_in  = f"// DW Input: {layer_name}, shape={list(in_d.shape)}, sc={in_s:.8f}, zp={in_zp}"
    hdr_out = f"// DW Output: {layer_name}, shape={list(out_d.shape)}, sc={out_s:.8f}, zp={out_zp}"
    write_hex_file(os.path.join(layer_dir, "input.hex.txt"),    in_d,  hdr_in)
    write_int_file(os.path.join(layer_dir, "input.int.txt"),    in_d,  hdr_in)
    write_hex_file(os.path.join(layer_dir, "output.hex.txt"),   out_d, hdr_out)
    write_int_file(os.path.join(layer_dir, "output.int.txt"),   out_d, hdr_out)
    write_float_file(os.path.join(layer_dir, "output.float.txt"), out_d, out_s, out_zp, hdr_out)
    dump_weight_files(layer_dir, "kernel_dw", dw_kernel_tensor,
                      layout_note="[1, KH, KW, Cout]  TFLite DW layout")
    if bias_tensor is not None:
        write_bias_int(os.path.join(layer_dir, "bias.int.txt"),
                       bias_tensor['data'], f"// DW bias {layer_name}: int32")

    # params.json
    params = {
        "layer_type": "DepthwiseConv2D",
        "layer_name": layer_name,
        "input": {"shape": list(in_d.shape), "scale": in_s, "zero_point": in_zp},
        "kernel": {"shape": list(dw_d.shape), "layout": "[1, KH, KW, Cout]"},
        "kernel_scales": [float(s) for s in dw_scales],
        "bias_present": bias_tensor is not None,
        "padding": padding_mode,
        "output": {"shape": list(out_d.shape), "scale": out_s, "zero_point": out_zp},
    }
    with open(os.path.join(layer_dir, "params.json"), 'w') as fp:
        json.dump(params, fp, indent=2)

    print(f"  [{layer_num:02d}] DW {layer_name:<40} in={list(in_d.shape)} → out={list(out_d.shape)}")
    return layer_dir


def write_algo_pointwise_conv(out_dir, layer_num, layer_name,
                               in_tensor, pw_kernel_tensor, bias_tensor,
                               out_tensor, has_relu=False):
    """
    Viết ALGO.txt và dump files cho PointwiseConv2D (1×1 Conv2D).
    Là bước thứ hai của SeparableConv2D.
    """
    layer_dir = os.path.join(out_dir, f"layer_{layer_num:02d}_{safe_name(layer_name)}_PW")
    os.makedirs(layer_dir, exist_ok=True)

    in_s   = in_tensor['scale']
    in_zp  = in_tensor['zp']
    in_d   = in_tensor['data']

    pw_d      = pw_kernel_tensor['data']       # shape: (Cout, 1, 1, Cin)
    pw_scales = pw_kernel_tensor['scales_arr']

    bias_d = bias_tensor['data'] if bias_tensor else None

    out_d  = out_tensor['data']
    out_s  = out_tensor['scale']
    out_zp = out_tensor['zp']

    N, H, W, Cin  = in_d.shape
    Cout          = pw_d.shape[0]

    with open(os.path.join(layer_dir, "ALGO.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write(f"LAYER {layer_num:02d}: POINTWISE CONV2D (1×1) — {layer_name}\n")
        f.write("=" * 72 + "\n\n")
        f.write("MÔ TẢ:\n")
        f.write("  PointwiseConv2D = Conv2D với kernel 1×1.\n")
        f.write("  Trộn thông tin giữa các channels tại mỗi spatial position.\n")
        f.write("  Là bước thứ hai của SeparableConv2D (sau DepthwiseConv2D).\n\n")
        f.write("SPECS:\n")
        f.write(f"  Input  : shape={list(in_d.shape)}, scale={in_s:.10f}, zp={in_zp}\n")
        f.write(f"  Kernel : shape={list(pw_d.shape)}  [Cout, 1, 1, Cin]\n")
        f.write(f"  Cout={Cout}, Cin={Cin}\n")
        f.write(f"  Output : shape={list(out_d.shape)}, scale={out_s:.10f}, zp={out_zp}\n")
        f.write(f"  ReLU   : {'YES (fused)' if has_relu else 'NO'}\n\n")

        f.write("GIẢI THÍCH CÁC BIẾN:\n")
        f.write("  ┌─────────────┬──────────┬────────────────────────────────────────────┐\n")
        f.write("  │ Biến        │ Kiểu     │ Ý nghĩa                                    │\n")
        f.write("  ├─────────────┼──────────┼────────────────────────────────────────────┤\n")
        f.write("  │ n           │ int      │ batch index (luôn = 0 với ASIC 1 ảnh)      │\n")
        f.write(f"  │ H, W        │ int      │ spatial size input = {H}×{W}              │\n")
        f.write(f"  │ Cin         │ int      │ input channels = {Cin}                    │\n")
        f.write(f"  │ Cout        │ int      │ output channels = {Cout}                  │\n")
        f.write("  │ h, w        │ int      │ spatial position đang tính (0..H-1, 0..W-1)│\n")
        f.write("  │ ci          │ int      │ input channel index đang cộng vào acc      │\n")
        f.write("  │ co          │ int      │ output channel index đang tính             │\n")
        f.write("  │ x_int8      │ int8     │ 1 activation tại [n, h, w, ci]             │\n")
        f.write("  │ w_int8      │ int8     │ 1 weight tại kernel[co, 0, 0, ci]          │\n")
        f.write("  │ x_zp        │ int8     │ zero-point của input tensor                │\n")
        f.write("  │ acc         │ int32    │ accumulator — tổng Cin tích (x-zp)×w       │\n")
        f.write("  │ bias[co]    │ int32    │ bias sau fold BN, 1 giá trị/output channel │\n")
        f.write("  │ S_in        │ float32  │ scale của input tensor                     │\n")
        f.write("  │ S_weight[co]│ float32  │ scale của weight channel co (per-channel)  │\n")
        f.write("  │ S_out       │ float32  │ scale của output tensor                    │\n")
        f.write("  │ M_int32     │ int32    │ multiplier requant (Q1.30)                 │\n")
        f.write("  │ total_shift │ int      │ số bit shift phải sau nhân M_int32         │\n")
        f.write("  │ out_zp      │ int8     │ zero-point output                          │\n")
        f.write("  │ out_q       │ int8     │ output sau requant + clip                  │\n")
        f.write("  │ out_val     │ int8     │ output cuối (= out_q hoặc ReLU(out_q))     │\n")
        f.write("  └─────────────┴──────────┴────────────────────────────────────────────┘\n\n")

        f.write("VỊ TRÍ TRONG BỘ NHỚ / FILE:\n")
        f.write(f"  INPUT  [n=0, h, w, ci]     →  offset = h×{W*Cin} + w×{Cin} + ci\n")
        f.write(f"  OUTPUT [n=0, h, w, co]     →  offset = h×{W*Cout} + w×{Cout} + co\n")
        f.write(f"  KERNEL [co, 0, 0, ci]      →  offset = co×{Cin} + ci\n")
        f.write(f"    (kernel 4D [Cout,1,1,Cin] flatten = kernel 2D [Cout,Cin] row-major)\n")
        f.write(f"  BIAS   [co]                →  offset = co\n\n")
        f.write("  Lưu ý: PW không có vòng lặp KH/KW vì kernel = 1×1.\n")
        f.write("         Mỗi spatial position (h,w) tính độc lập → rất dễ pipeline.\n\n")

        f.write("MINH HOẠ: output[h=2, w=3, co=0] được tính từ:\n")
        f.write(f"  input[0, 2, 3, ci=0..{Cin-1}]  (1 vector Cin={Cin} elements)\n")
        f.write(f"  kernel[co=0, 0, 0, ci=0..{Cin-1}] (hàng 0 của weight matrix)\n")
        f.write(f"  = dot product của 2 vector {Cin}-chiều\n\n")

        f.write("THUẬT TOÁN INTEGER-ONLY:\n")
        f.write("  // Với mỗi [n, h, w, co]:\n")
        f.write("  int32 acc = 0\n")
        f.write("  for ci in 0..Cin-1:\n")
        f.write("    x_int8 = input[n, h, w, ci]           // int8\n")
        f.write("    w_int8 = kernel[co, 0, 0, ci]         // int8\n")
        f.write("    acc   += (int32)(x_int8 - x_zp) × (int32)(w_int8)\n")
        if bias_d is not None:
            f.write("  acc += bias[co]                           // int32\n")
        f.write("\n  // Requantization:\n")
        f.write("  M_float  = S_in × S_weight[co] / S_out\n")
        f.write("  M_int32, total_shift  (xem bảng bên dưới)\n")
        f.write("  int64 scaled  = (int64)acc × (int64)M_int32\n")
        f.write("  int32 shifted = scaled >>> total_shift\n")
        f.write("  int8  out_q   = clip(shifted + out_zp, -128, 127)\n")
        if has_relu:
            f.write("  int8  out_val = max(out_q, out_zp)   // ReLU (so sánh với zp, không phải 0)\n\n")
        else:
            f.write("\n")
        f.write("LƯU Ý BIAS:\n")
        f.write("  Layer này dùng use_bias=False → bias tensor = zeros.\n")
        f.write("  Bước cộng bias (acc += bias[co]) không đóng góp gì.\n")
        f.write("  RTL có thể bỏ qua bước cộng bias cho DW/PW layers.\n\n")

        f.write("REQUANTIZATION PARAMS:\n")
        f.write(f"  S_in  = {in_s:.12f}\n")
        f.write(f"  S_out = {out_s:.12f}\n")
        f.write(f"  {'co':>4}  {'S_w':>14}  {'M_float':>18}  {'M_int32':>12}  {'shift':>6}\n")
        f.write(f"  {'─'*62}\n")
        for co in range(min(Cout, len(pw_scales))):
            sw = float(pw_scales[co])
            M  = in_s * sw / out_s
            shift, M_norm = 0, M
            while M_norm < 0.5 and shift < 31:
                M_norm *= 2; shift += 1
            M_i32 = int(round(M_norm * 2**30))
            f.write(f"  {co:>4}  {sw:>14.10f}  {M:>18.12f}  {M_i32:>12d}  {shift+30:>6d}\n")

        f.write(f"\nVerilog TEMPLATE:\n")
        f.write(f"  wire signed [63:0] scaled  = $signed(acc) * $signed(M_int32[co]);\n")
        f.write(f"  wire signed [31:0] shifted = scaled >>> SHIFT[co];\n")
        f.write(f"  wire signed  [7:0] out_q   = clip(shifted + {out_zp}, -128, 127);\n")
        if has_relu:
            f.write(f"  wire signed  [7:0] out_val = (out_q < {out_zp}) ? 8'sd{out_zp} : out_q;\n")

    hdr_in  = f"// PW Input : {layer_name}, shape={list(in_d.shape)}, sc={in_s:.8f}, zp={in_zp}"
    hdr_out = f"// PW Output: {layer_name}, shape={list(out_d.shape)}, sc={out_s:.8f}, zp={out_zp}"
    write_hex_file(os.path.join(layer_dir, "input.hex.txt"),    in_d,  hdr_in)
    write_int_file(os.path.join(layer_dir, "input.int.txt"),    in_d,  hdr_in)
    write_hex_file(os.path.join(layer_dir, "output.hex.txt"),   out_d, hdr_out)
    write_int_file(os.path.join(layer_dir, "output.int.txt"),   out_d, hdr_out)
    write_float_file(os.path.join(layer_dir, "output.float.txt"), out_d, out_s, out_zp, hdr_out)
    dump_weight_files(layer_dir, "kernel_pw", pw_kernel_tensor,
                      layout_note="[Cout, 1, 1, Cin]  TFLite PW layout")
    if bias_tensor is not None:
        write_bias_int(os.path.join(layer_dir, "bias.int.txt"),
                       bias_tensor['data'], f"// PW bias {layer_name}: int32")

    params = {
        "layer_type": "PointwiseConv2D (1x1 Conv2D)",
        "layer_name": layer_name,
        "input":  {"shape": list(in_d.shape), "scale": in_s, "zero_point": in_zp},
        "kernel": {"shape": list(pw_d.shape), "layout": "[Cout, 1, 1, Cin]"},
        "kernel_scales": [float(s) for s in pw_scales],
        "bias_present": bias_tensor is not None,
        "output": {"shape": list(out_d.shape), "scale": out_s, "zero_point": out_zp},
    }
    with open(os.path.join(layer_dir, "params.json"), 'w') as fp:
        json.dump(params, fp, indent=2)

    print(f"  [{layer_num:02d}] PW {layer_name:<40} in={list(in_d.shape)} → out={list(out_d.shape)}")
    return layer_dir


def write_algo_maxpool(out_dir, layer_num, layer_name,
                       in_tensor, out_tensor, pool_size=2, stride=2):
    """MaxPooling2D — integer passthrough, không cần requant"""
    layer_dir = os.path.join(out_dir, f"layer_{layer_num:02d}_{safe_name(layer_name)}")
    os.makedirs(layer_dir, exist_ok=True)

    in_s   = in_tensor['scale']
    in_zp  = in_tensor['zp']
    in_d   = in_tensor['data']
    out_d  = out_tensor['data']
    out_s  = out_tensor['scale']
    out_zp = out_tensor['zp']

    with open(os.path.join(layer_dir, "ALGO.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write(f"LAYER {layer_num:02d}: MAXPOOLING2D — {layer_name}\n")
        f.write("=" * 72 + "\n\n")
        f.write("MÔ TẢ:\n")
        f.write("  MaxPool chọn giá trị lớn nhất trong mỗi cửa sổ pool_size×pool_size.\n")
        f.write("  QUAN TRỌNG: MaxPool hoạt động trực tiếp trên int8 — KHÔNG cần\n")
        f.write("  requantization vì scale/zp không thay đổi.\n")
        f.write("  So sánh int8 là monotone (order-preserving) nên max(int8) = max(float).\n\n")
        f.write("SPECS:\n")
        f.write(f"  Input     : shape={list(in_d.shape)}, scale={in_s:.10f}, zp={in_zp}\n")
        f.write(f"  Pool size : {pool_size}×{pool_size}\n")
        f.write(f"  Stride    : {stride}\n")
        f.write(f"  Padding   : VALID (không pad)\n")
        f.write(f"  Output    : shape={list(out_d.shape)}, scale={out_s:.10f}, zp={out_zp}\n\n")

        _, IH, IW, C = in_d.shape
        OH, OW = IH // stride, IW // stride
        f.write("GIẢI THÍCH CÁC BIẾN:\n")
        f.write("  ┌─────────────┬──────────┬────────────────────────────────────────────┐\n")
        f.write("  │ Biến        │ Kiểu     │ Ý nghĩa                                    │\n")
        f.write("  ├─────────────┼──────────┼────────────────────────────────────────────┤\n")
        f.write(f"  │ IH, IW      │ int      │ input height={IH}, width={IW}             │\n")
        f.write(f"  │ OH, OW      │ int      │ output height={OH}, width={OW}            │\n")
        f.write(f"  │ C           │ int      │ channels = {C} (giống nhau input/output)  │\n")
        f.write(f"  │ pool_size   │ int      │ kích thước cửa sổ = {pool_size}           │\n")
        f.write(f"  │ stride      │ int      │ bước nhảy = {stride}                      │\n")
        f.write("  │ oh, ow      │ int      │ output row, col đang tính                  │\n")
        f.write("  │ ph, pw      │ int      │ vị trí trong cửa sổ pool (0..pool_size-1)  │\n")
        f.write("  │ ih, iw      │ int      │ input position = oh×stride+ph, ow×stride+pw│\n")
        f.write("  │ ch          │ int      │ channel index (xử lý độc lập mỗi channel)  │\n")
        f.write("  │ max_val     │ int8     │ giá trị lớn nhất tìm được trong cửa sổ     │\n")
        f.write("  └─────────────┴──────────┴────────────────────────────────────────────┘\n\n")

        f.write("VỊ TRÍ TRONG BỘ NHỚ / FILE:\n")
        f.write(f"  INPUT  [n=0, ih, iw, ch]  →  offset = ih×{IW*C} + iw×{C} + ch\n")
        f.write(f"  OUTPUT [n=0, oh, ow, ch]  →  offset = oh×{OW*C} + ow×{C} + ch\n\n")
        f.write(f"  Mỗi output [oh, ow, ch] chỉ đọc {pool_size*pool_size} pixels từ input:\n")
        f.write(f"    input[oh×{stride}+0..{pool_size-1}, ow×{stride}+0..{pool_size-1}, ch]\n\n")

        f.write("THUẬT TOÁN (int8 arithmetic):\n")
        f.write("  // Với mỗi [n, oh, ow, ch]:\n")
        f.write("  int8 max_val = INT8_MIN  // = -128\n")
        f.write("  for ph in 0..pool_size-1:\n")
        f.write("    for pw in 0..pool_size-1:\n")
        f.write("      ih = oh × stride + ph\n")
        f.write("      iw = ow × stride + pw\n")
        f.write("      if input[n, ih, iw, ch] > max_val:\n")
        f.write("        max_val = input[n, ih, iw, ch]   // signed int8 comparison\n")
        f.write("  output[n, oh, ow, ch] = max_val\n\n")
        f.write("KHÔNG CÓ REQUANTIZATION:\n")
        f.write(f"  in_scale  = out_scale = {in_s:.10f}\n")
        f.write(f"  in_zp     = out_zp    = {in_zp}\n\n")
        f.write("Verilog TEMPLATE:\n")
        f.write("  // Signed comparison: dùng $signed() hoặc wire signed [7:0]\n")
        f.write("  wire signed [7:0] a, b;\n")
        f.write("  assign max_ab = ($signed(a) > $signed(b)) ? a : b;\n")

    hdr_in  = f"// MaxPool Input : {layer_name}, shape={list(in_d.shape)}, sc={in_s:.8f}, zp={in_zp}"
    hdr_out = f"// MaxPool Output: {layer_name}, shape={list(out_d.shape)}, sc={out_s:.8f}, zp={out_zp}"
    write_hex_file(os.path.join(layer_dir, "input.hex.txt"),    in_d,  hdr_in)
    write_int_file(os.path.join(layer_dir, "input.int.txt"),    in_d,  hdr_in)
    write_hex_file(os.path.join(layer_dir, "output.hex.txt"),   out_d, hdr_out)
    write_int_file(os.path.join(layer_dir, "output.int.txt"),   out_d, hdr_out)
    write_float_file(os.path.join(layer_dir, "output.float.txt"), out_d, out_s, out_zp, hdr_out)

    params = {
        "layer_type": "MaxPooling2D",
        "layer_name": layer_name,
        "pool_size": pool_size,
        "stride": stride,
        "padding": "VALID",
        "no_requantization": True,
        "input":  {"shape": list(in_d.shape), "scale": in_s, "zero_point": in_zp},
        "output": {"shape": list(out_d.shape), "scale": out_s, "zero_point": out_zp},
    }
    with open(os.path.join(layer_dir, "params.json"), 'w') as fp:
        json.dump(params, fp, indent=2)

    print(f"  [{layer_num:02d}] MaxPool {layer_name:<38} in={list(in_d.shape)} → out={list(out_d.shape)}")
    return layer_dir


def write_algo_gap(out_dir, layer_num, layer_name, in_tensor, out_tensor):
    """GlobalAveragePooling2D"""
    layer_dir = os.path.join(out_dir, f"layer_{layer_num:02d}_{safe_name(layer_name)}")
    os.makedirs(layer_dir, exist_ok=True)

    in_s   = in_tensor['scale']
    in_zp  = in_tensor['zp']
    in_d   = in_tensor['data']
    out_d  = out_tensor['data']
    out_s  = out_tensor['scale']
    out_zp = out_tensor['zp']

    N, H, W, C = in_d.shape

    import math
    M  = in_s / out_s
    shift_r, M_norm = 0, M
    while M_norm < 0.5 and shift_r < 31:
        M_norm *= 2; shift_r += 1
    M_i32 = int(round(M_norm * 2**30))

    with open(os.path.join(layer_dir, "ALGO.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write(f"LAYER {layer_num:02d}: GLOBAL AVERAGE POOLING 2D — {layer_name}\n")
        f.write("=" * 72 + "\n\n")
        f.write("MÔ TẢ:\n")
        f.write(f"  Tính trung bình của toàn bộ spatial map (H×W) cho mỗi channel.\n")
        f.write(f"  Input ({H}×{W}×{C}) → Output (1×1×{C}) = ({C},)\n\n")
        f.write("SPECS:\n")
        f.write(f"  Input  : shape={list(in_d.shape)}, scale={in_s:.10f}, zp={in_zp}\n")
        f.write(f"  H={H}, W={W}, C={C}, N_elements_per_ch = H×W = {H*W}\n")
        f.write(f"  Output : shape={list(out_d.shape)}, scale={out_s:.10f}, zp={out_zp}\n\n")

        import math
        f.write("GIẢI THÍCH CÁC BIẾN:\n")
        f.write("  ┌─────────────┬──────────┬────────────────────────────────────────────┐\n")
        f.write("  │ Biến        │ Kiểu     │ Ý nghĩa                                    │\n")
        f.write("  ├─────────────┼──────────┼────────────────────────────────────────────┤\n")
        f.write(f"  │ H, W        │ int      │ spatial size = {H}×{W}                    │\n")
        f.write(f"  │ C           │ int      │ channels = {C}                            │\n")
        f.write(f"  │ N_spatial   │ int      │ H×W = {H*W} (số pixels mỗi channel)       │\n")
        f.write("  │ ch          │ int      │ channel index đang tính trung bình          │\n")
        f.write("  │ h, w        │ int      │ vị trí spatial đang cộng vào sum            │\n")
        f.write("  │ sum         │ int32    │ tổng (x - in_zp) trên toàn H×W             │\n")
        f.write("  │ avg_adj     │ int32    │ sum / N_spatial (trung bình đã trừ zp)      │\n")
        f.write("  │ in_zp       │ int8     │ zero-point của input tensor                 │\n")
        f.write("  │ S_in        │ float32  │ scale của input tensor                      │\n")
        f.write("  │ S_out       │ float32  │ scale của output tensor                     │\n")
        f.write(f"  │ M_int32     │ int32    │ = {M_i32} (S_in/S_out trong Q1.30)         │\n")
        f.write(f"  │ total_shift │ int      │ = {shift_r+30}                             │\n")
        f.write(f"  │ out_zp      │ int8     │ = {out_zp}                                 │\n")
        f.write("  │ out_val     │ int8     │ output sau requant + clip [-128,127]         │\n")
        f.write("  └─────────────┴──────────┴────────────────────────────────────────────┘\n\n")

        f.write("VỊ TRÍ TRONG BỘ NHỚ / FILE:\n")
        f.write(f"  INPUT  [n=0, h, w, ch]  →  offset = h×{W*C} + w×{C} + ch\n")
        f.write(f"  OUTPUT [ch]             →  offset = ch  ({C} int8 values liên tiếp)\n\n")
        f.write(f"  Để tính output[ch], cộng tất cả {H*W} pixels của channel ch:\n")
        f.write(f"    input[0, 0..{H-1}, 0..{W-1}, ch]  stride {C} trong bộ nhớ nếu NHWC\n\n")
        acc_bits = int(np.ceil(np.log2(127*H*W + 1))) + 1
        f.write(f"  Accumulator cần ít nhất {acc_bits} bits signed\n")
        f.write(f"    (max: 127 × {H*W} = {127*H*W}, min: -128 × {H*W} = {-128*H*W})\n\n")

        f.write("THUẬT TOÁN INTEGER-ONLY:\n")
        f.write(f"  N_spatial = H × W = {H} × {W} = {H*W}\n")
        f.write("  // Với mỗi channel ch:\n")
        f.write("  int32 sum = 0\n")
        f.write("  for h in 0..H-1:\n")
        f.write("    for w in 0..W-1:\n")
        f.write("      sum += (int32)(input[0, h, w, ch] - in_zp)   // int32 accumulation\n")
        f.write(f"  // Chia lấy trung bình: dùng shift nếu N_spatial là power-of-2\n")
        if (H * W) & (H * W - 1) == 0:
            shift = int(math.log2(H * W))
            f.write(f"  // N_spatial = {H*W} = 2^{shift} → chia bằng shift\n")
            f.write(f"  int32 avg_adj = (sum + (1 << {shift-1})) >> {shift}  // +0.5 trước shift\n")
        else:
            f.write(f"  // N_spatial = {H*W} — không phải power-of-2, cần divider\n")
            f.write(f"  int32 avg_adj = round(sum / {H*W})\n")
        f.write("  // Requantization:\n")
        f.write("  M = S_in / S_out  (không có weight, không có per-channel scale)\n")
        f.write(f"  M = {in_s:.10f} / {out_s:.10f} = {M:.10f}\n")
        f.write(f"  M_int32 = {M_i32}, total_shift = {shift_r+30}\n")
        f.write(f"  int64 scaled  = (int64)avg_adj × (int64){M_i32}\n")
        f.write(f"  int32 shifted = scaled >>> {shift_r+30}\n")
        f.write(f"  int8  out_val = clip(shifted + {out_zp}, -128, 127)\n\n")
        f.write("Verilog NOTE:\n")
        f.write(f"  Cần accumulator {acc_bits}-bit signed để tránh overflow\n")
        f.write(f"  Max possible sum = 127 × {H*W} = {127*H*W}\n")

    hdr_in  = f"// GAP Input : shape={list(in_d.shape)}, sc={in_s:.8f}, zp={in_zp}"
    hdr_out = f"// GAP Output: shape={list(out_d.shape)}, sc={out_s:.8f}, zp={out_zp}"
    write_hex_file(os.path.join(layer_dir, "input.hex.txt"),    in_d,  hdr_in)
    write_int_file(os.path.join(layer_dir, "input.int.txt"),    in_d,  hdr_in)
    write_hex_file(os.path.join(layer_dir, "output.hex.txt"),   out_d, hdr_out)
    write_int_file(os.path.join(layer_dir, "output.int.txt"),   out_d, hdr_out)
    write_float_file(os.path.join(layer_dir, "output.float.txt"), out_d, out_s, out_zp, hdr_out)

    params = {
        "layer_type": "GlobalAveragePooling2D",
        "H": H, "W": W, "C": C, "N_spatial": H * W,
        "input":  {"shape": list(in_d.shape), "scale": in_s, "zero_point": in_zp},
        "output": {"shape": list(out_d.shape), "scale": out_s, "zero_point": out_zp},
        "M_int32": M_i32, "total_shift": shift_r + 30,
    }
    with open(os.path.join(layer_dir, "params.json"), 'w') as fp:
        json.dump(params, fp, indent=2)

    print(f"  [{layer_num:02d}] GAP   {layer_name:<40} in={list(in_d.shape)} → out={list(out_d.shape)}")
    return layer_dir


def write_algo_dense(out_dir, layer_num, layer_name,
                     in_tensor, kernel_tensor, bias_tensor,
                     out_tensor, has_relu=False):
    """Dense (Fully Connected) layer"""
    layer_dir = os.path.join(out_dir, f"layer_{layer_num:02d}_{safe_name(layer_name)}")
    os.makedirs(layer_dir, exist_ok=True)

    in_s   = in_tensor['scale']
    in_zp  = in_tensor['zp']
    in_d   = in_tensor['data']

    W_d      = kernel_tensor['data']       # shape: (Cout, Cin)
    W_scales = kernel_tensor['scales_arr']

    bias_d = bias_tensor['data'] if bias_tensor else None

    out_d  = out_tensor['data']
    out_s  = out_tensor['scale']
    out_zp = out_tensor['zp']

    Cout, Cin = W_d.shape

    with open(os.path.join(layer_dir, "ALGO.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write(f"LAYER {layer_num:02d}: DENSE (FULLY CONNECTED) — {layer_name}\n")
        f.write("=" * 72 + "\n\n")
        f.write("MÔ TẢ:\n")
        f.write(f"  Cin={Cin} inputs → Cout={Cout} outputs.\n")
        f.write(f"  Mỗi output neuron: dot product với 1 hàng của weight matrix.\n\n")
        f.write("SPECS:\n")
        f.write(f"  Input  : shape={list(in_d.shape)}, scale={in_s:.10f}, zp={in_zp}\n")
        f.write(f"  Weight : shape={list(W_d.shape)}  [Cout={Cout}, Cin={Cin}]\n")
        f.write(f"  Output : shape={list(out_d.shape)}, scale={out_s:.10f}, zp={out_zp}\n")
        f.write(f"  ReLU   : {'YES (fused)' if has_relu else 'NO'}\n\n")

        f.write("GIẢI THÍCH CÁC BIẾN:\n")
        f.write("  ┌─────────────┬──────────┬────────────────────────────────────────────┐\n")
        f.write("  │ Biến        │ Kiểu     │ Ý nghĩa                                    │\n")
        f.write("  ├─────────────┼──────────┼────────────────────────────────────────────┤\n")
        f.write(f"  │ Cin         │ int      │ số input neurons = {Cin}                  │\n")
        f.write(f"  │ Cout        │ int      │ số output neurons = {Cout}                │\n")
        f.write("  │ ci          │ int      │ index input neuron đang nhân               │\n")
        f.write("  │ co          │ int      │ index output neuron đang tính              │\n")
        f.write("  │ x_int8      │ int8     │ 1 input activation tại [ci]                │\n")
        f.write("  │ w_int8      │ int8     │ 1 weight tại weight[co, ci]                │\n")
        f.write("  │ x_zp        │ int8     │ zero-point của input tensor                │\n")
        f.write("  │ acc         │ int32    │ accumulator — dot product (x-zp)·w         │\n")
        f.write("  │ bias[co]    │ int32    │ bias của output neuron co                  │\n")
        f.write("  │ S_in        │ float32  │ scale của input tensor                     │\n")
        f.write("  │ S_weight[co]│ float32  │ scale của hàng co trong weight matrix      │\n")
        f.write("  │ S_out       │ float32  │ scale của output tensor                    │\n")
        f.write("  │ M_int32     │ int32    │ multiplier requant (Q1.30)                 │\n")
        f.write("  │ total_shift │ int      │ số bit shift phải sau nhân M_int32         │\n")
        f.write("  │ out_zp      │ int8     │ zero-point output                          │\n")
        f.write("  │ out_q       │ int8     │ output sau requant + clip                  │\n")
        f.write("  │ out_val     │ int8     │ output cuối (= out_q hoặc ReLU(out_q))     │\n")
        f.write("  └─────────────┴──────────┴────────────────────────────────────────────┘\n\n")

        f.write("VỊ TRÍ TRONG BỘ NHỚ / FILE:\n")
        f.write(f"  INPUT   [ci]        →  offset = ci          ({Cin} int8 values liên tiếp)\n")
        f.write(f"  OUTPUT  [co]        →  offset = co          ({Cout} int8 values liên tiếp)\n")
        f.write(f"  WEIGHT  [co, ci]    →  offset = co×{Cin} + ci  (row-major, hàng co đọc liên tiếp)\n")
        f.write(f"  BIAS    [co]        →  offset = co          ({Cout} int32 values liên tiếp)\n\n")
        f.write("  Cách đọc weight_fc.hex.txt:\n")
        f.write(f"    Dòng 0..{Cin-1}      → weight[co=0, ci=0..{Cin-1}]  (hàng 0 = neuron 0)\n")
        f.write(f"    Dòng {Cin}..{2*Cin-1}  → weight[co=1, ci=0..{Cin-1}]  (hàng 1 = neuron 1)\n")
        f.write(f"    ...  (tổng {Cout}×{Cin}={Cout*Cin} dòng)\n\n")
        f.write(f"  Để tính output[co], đọc {Cin} weights liên tiếp từ offset co×{Cin}.\n\n")

        f.write("THUẬT TOÁN INTEGER-ONLY:\n")
        f.write("  // Với mỗi output neuron co:\n")
        f.write("  int32 acc = 0\n")
        f.write("  for ci in 0..Cin-1:\n")
        f.write("    x_int8 = input[ci]               // int8\n")
        f.write("    w_int8 = weight[co, ci]           // int8\n")
        f.write("    acc   += (int32)(x_int8 - x_zp) × (int32)(w_int8)\n")
        if bias_d is not None:
            f.write("  acc += bias[co]                     // int32\n")
        f.write("\n  // Requantization:\n")
        f.write("  M = S_in × S_weight[co] / S_out\n")
        f.write("  int64 scaled  = (int64)acc × (int64)M_int32\n")
        f.write("  int32 shifted = scaled >>> total_shift\n")
        f.write("  int8  out_q   = clip(shifted + out_zp, -128, 127)\n")
        if has_relu:
            f.write(f"  int8  out_val = max(out_q, out_zp)   // ReLU\n\n")
        else:
            f.write("\n")

        f.write("REQUANTIZATION PARAMS:\n")
        f.write(f"  S_in  = {in_s:.12f}\n")
        f.write(f"  S_out = {out_s:.12f}\n")
        f.write(f"  {'co':>4}  {'S_w':>14}  {'M_float':>18}  {'M_int32':>12}  {'shift':>6}\n")
        f.write(f"  {'─'*62}\n")
        for co in range(min(Cout, len(W_scales))):
            sw = float(W_scales[co])
            M  = in_s * sw / out_s
            shift, M_norm = 0, M
            while M_norm < 0.5 and shift < 31:
                M_norm *= 2; shift += 1
            M_i32 = int(round(M_norm * 2**30))
            f.write(f"  {co:>4}  {sw:>14.10f}  {M:>18.12f}  {M_i32:>12d}  {shift+30:>6d}\n")

        f.write(f"\nOPERATION COUNT:\n")
        f.write(f"  MACs per output = Cin = {Cin}\n")
        f.write(f"  Total MACs      = Cout × Cin = {Cout} × {Cin} = {Cout*Cin}\n")
        f.write(f"\nVerilog TEMPLATE:\n")
        f.write(f"  wire signed [63:0] scaled  = $signed(acc) * $signed(M_int32[co]);\n")
        f.write(f"  wire signed [31:0] shifted = scaled >>> SHIFT[co];\n")
        f.write(f"  wire signed  [7:0] out_q   = clip(shifted + {out_zp}, -128, 127);\n")
        if has_relu:
            f.write(f"  wire signed  [7:0] out_val = (out_q < {out_zp}) ? 8'sd{out_zp} : out_q;\n")

    hdr_in  = f"// Dense Input : {layer_name}, shape={list(in_d.shape)}, sc={in_s:.8f}, zp={in_zp}"
    hdr_out = f"// Dense Output: {layer_name}, shape={list(out_d.shape)}, sc={out_s:.8f}, zp={out_zp}"
    write_hex_file(os.path.join(layer_dir, "input.hex.txt"),    in_d,  hdr_in)
    write_int_file(os.path.join(layer_dir, "input.int.txt"),    in_d,  hdr_in)
    write_hex_file(os.path.join(layer_dir, "output.hex.txt"),   out_d, hdr_out)
    write_int_file(os.path.join(layer_dir, "output.int.txt"),   out_d, hdr_out)
    write_float_file(os.path.join(layer_dir, "output.float.txt"), out_d, out_s, out_zp, hdr_out)
    dump_weight_files(layer_dir, "weight_fc",  kernel_tensor,
                      layout_note=f"[Cout={Cout}, Cin={Cin}]  Dense row-major")
    if bias_tensor is not None:
        write_bias_int(os.path.join(layer_dir, "bias.int.txt"),
                       bias_tensor['data'], f"// Dense bias {layer_name}: {Cout} neurons, int32")

    params = {
        "layer_type": "Dense (FullyConnected)",
        "layer_name": layer_name,
        "Cin": Cin, "Cout": Cout, "total_MACs": Cin * Cout,
        "relu_fused": has_relu,
        "input":  {"shape": list(in_d.shape), "scale": in_s, "zero_point": in_zp},
        "kernel": {"shape": list(W_d.shape), "layout": "[Cout, Cin] row-major"},
        "kernel_scales": [float(s) for s in W_scales],
        "bias_present": bias_tensor is not None,
        "output": {"shape": list(out_d.shape), "scale": out_s, "zero_point": out_zp},
    }
    with open(os.path.join(layer_dir, "params.json"), 'w') as fp:
        json.dump(params, fp, indent=2)

    print(f"  [{layer_num:02d}] Dense {layer_name:<39} in={list(in_d.shape)} → out={list(out_d.shape)}")
    return layer_dir


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: PIPELINE OVERVIEW + VERIFICATION GUIDE
# ═════════════════════════════════════════════════════════════════════════════

def write_pipeline_overview(out_dir, model_path, logit, prob, out_raw_int8,
                             inp_det, out_det, layer_dirs):
    """Tạo file sơ đồ toàn bộ pipeline integer-only"""
    p = os.path.join(out_dir, "PIPELINE_OVERVIEW.txt")
    out_s  = float(out_det['quantization_parameters']['scales'][0])
    out_zp = int(out_det['quantization_parameters']['zero_points'][0])
    inp_s  = float(inp_det['quantization_parameters']['scales'][0])
    inp_zp = int(inp_det['quantization_parameters']['zero_points'][0])

    label  = "ACTIVE 😊" if prob >= ACTIVE_THRESHOLD else "FATIGUE 😴"

    with open(p, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DROWSINESS ASIC — INTEGER-ONLY INFERENCE PIPELINE OVERVIEW\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL  : " + model_path + "\n")
        f.write(f"RESULT : logit={logit:.6f}, prob_active={prob:.6f}\n")
        f.write(f"         out_raw_int8={int(out_raw_int8)}\n")
        f.write(f"LABEL  : {label}  (threshold={ACTIVE_THRESHOLD:.0%})\n\n")

        f.write("─" * 80 + "\n")
        f.write("PIPELINE (integer-only, dùng int8 activation + int32 accumulation):\n")
        f.write("─" * 80 + "\n\n")
        f.write(f"  [PIXEL Q0.7]  int8(128×128)\n")
        f.write(f"       ↓  Requantize (scale/zp convert)\n")
        f.write(f"  [INPUT]       int8(1×128×128×1)   scale={inp_s:.8f}, zp={inp_zp}\n")
        f.write(f"       ↓  SepConv(32, 3×3, no_bias): DW→ReLU + PW→ReLU\n")
        f.write(f"  [Block1 out]  int8(1×128×128×32)\n")
        f.write(f"       ↓  MaxPool(2×2, s=2)\n")
        f.write(f"  [Pool1 out]   int8(1×64×64×32)\n")
        f.write(f"       ↓  SepConv(64, 3×3, no_bias): DW→ReLU + PW→ReLU\n")
        f.write(f"  [Block2 out]  int8(1×64×64×64)\n")
        f.write(f"       ↓  MaxPool(2×2, s=2)\n")
        f.write(f"  [Pool2 out]   int8(1×32×32×64)\n")
        f.write(f"       ↓  SepConv(128, 3×3, no_bias): DW→ReLU + PW→ReLU\n")
        f.write(f"  [Block3 out]  int8(1×32×32×128)\n")
        f.write(f"       ↓  MaxPool(2×2, s=2)\n")
        f.write(f"  [Pool3 out]   int8(1×16×16×128)\n")
        f.write(f"       ↓  SepConv(128, 3×3, no_bias): DW→ReLU + PW→ReLU\n")
        f.write(f"  [Block4 out]  int8(1×16×16×128)\n")
        f.write(f"       ↓  GlobalAvgPool (H=16, W=16, N_spatial=256)\n")
        f.write(f"  [GAP out]     int8(128,)\n")
        f.write(f"       ↓  Dense(Cin=128, Cout=32) + ReLU\n")
        f.write(f"  [FC1 out]     int8(32,)\n")
        f.write(f"       ↓  Dense(Cin=32, Cout=1)   — logit output\n")
        f.write(f"  [Output]      int8(1,)            scale={out_s:.8f}, zp={out_zp}\n")
        f.write(f"       ↓  Dequantize + Sigmoid (có thể làm ngoài ASIC)\n")
        f.write(f"  [Decision]    prob={prob:.6f} → {'ACTIVE' if prob>=ACTIVE_THRESHOLD else 'FATIGUE'}\n\n")

        f.write("─" * 80 + "\n")
        f.write("REQUANTIZATION FORMULA (chung cho mọi Conv/Dense):\n")
        f.write("─" * 80 + "\n")
        f.write("""
  INPUT : int8 x, scale S_x, zero_point ZP_x
  WEIGHT: int8 w, scale S_w[ch] (per-channel), zero_point ZP_w=0

  STEP 1 — MAC (trong int32):
    acc = Σ (int32)(x[i] - ZP_x) × (int32)(w[i])  +  bias[ch]  (int32)

  STEP 2 — Compute M (một lần, offline):
    M = S_x × S_w[ch] / S_out
    Normalize: M_int32, total_shift  sao cho  M ≈ M_int32 / 2^total_shift
    (xem params.json trong từng layer folder)

  STEP 3 — Apply M (trong int64):
    int64 scaled  = (int64)acc × (int64)M_int32
    int32 shifted = scaled >>> total_shift         // arithmetic right shift

  STEP 4 — Requantize output:
    int8 out = clip(shifted + ZP_out, -128, 127)

  STEP 5 — ReLU (nếu có, sau requantize):
    int8 out = max(out, ZP_out)   // so sánh với ZP_out (không phải 0!)

  NOTE: ZP_w = 0 cho TFLite per-channel quantized weights → bỏ qua offset weight
""")

        f.write("─" * 80 + "\n")
        f.write("LAYER FOLDERS SINH RA:\n")
        f.write("─" * 80 + "\n")
        for d in layer_dirs:
            f.write(f"  {os.path.basename(d)}/\n")
        f.write("\nMỗi folder chứa:\n")
        f.write("  ALGO.txt         — Mô tả thuật toán, công thức, requant params\n")
        f.write("  input.hex.txt    — Input ($readmemh format)\n")
        f.write("  input.int.txt    — Input decimal\n")
        f.write("  output.hex.txt   — Golden output ($readmemh)\n")
        f.write("  output.int.txt   — Golden output decimal\n")
        f.write("  output.float.txt — Dequantized float (debug)\n")
        f.write("  kernel.hex.txt   — Weight hex (nếu có)\n")
        f.write("  bias.int.txt     — Bias int32 (nếu có)\n")
        f.write("  params.json      — Tất cả hyperparameters (machine-readable)\n")

    print(f"  💾 {p}")
    return p


def write_verification_guide(out_dir, logit, prob, out_raw_int8, out_det):
    """Hướng dẫn RTL engineer sử dụng file này"""
    out_s  = float(out_det['quantization_parameters']['scales'][0])
    out_zp = int(out_det['quantization_parameters']['zero_points'][0])
    label  = "ACTIVE" if prob >= ACTIVE_THRESHOLD else "FATIGUE"

    p = os.path.join(out_dir, "VERIFICATION_GUIDE.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HƯỚNG DẪN VERIFICATION CHO RTL ENGINEER\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MỤC TIÊU:\n")
        f.write("   RTL simulation phải cho kết quả MATCH với golden reference trong\n")
        f.write("   các file *.int.txt / *.hex.txt của từng layer.\n\n")

        f.write("2. LOAD INPUT VÀO TESTBENCH:\n")
        f.write("   $readmemh(\"layer_00_INPUT/input.hex.txt\", pixel_mem);\n")
        f.write("   Format: 1 hex byte/dòng, two's complement int8\n\n")

        f.write("3. LOAD WEIGHTS:\n")
        f.write("   Mỗi layer folder có kernel.hex.txt:\n")
        f.write("   $readmemh(\"layer_XX_<name>/kernel.hex.txt\", weight_mem);\n")
        f.write("   TFLite layout:\n")
        f.write("     DW kernel   : [1, KH, KW, Cout] — ch-major\n")
        f.write("     PW kernel   : [Cout, 1, 1, Cin] — row-major\n")
        f.write("     Dense kernel: [Cout, Cin]         — row-major\n\n")

        f.write("4. KIỂM TRA TỪNG LAYER:\n")
        f.write("   Sau khi layer RTL xử lý xong, dump output → so sánh với:\n")
        f.write("     output.int.txt  (so sánh decimal)\n")
        f.write("   hoặc:\n")
        f.write("     output.hex.txt  (so sánh hex)\n\n")

        f.write("5. TOLERANCE:\n")
        f.write("   TargET: 0 LSB (exact match)\n")
        f.write("   Chấp nhận: ±1 LSB (do rounding khác nhau giữa Python và RTL)\n")
        f.write("   KHÔNG chấp nhận: >1 LSB — cần debug ngay\n\n")

        f.write("6. FINAL OUTPUT CHECK:\n")
        f.write(f"   Expected out_raw_int8 = {int(out_raw_int8)}\n")
        f.write(f"   Expected logit        = {logit:.6f}\n")
        f.write(f"   Expected prob_active  = {prob:.6f}\n")
        f.write(f"   Expected label        = {label}\n")
        f.write(f"   out_scale = {out_s:.10f}, out_zp = {out_zp}\n")
        f.write(f"   Dequant: logit = (out_int8 - {out_zp}) × {out_s:.10f}\n\n")

        f.write("7. ĐIỂM HAY GÂY LỖI:\n")
        f.write("   a) Rounding: Python dùng round-half-to-even, RTL thường round-half-up\n")
        f.write("      → sai ±1 LSB tại biên → chấp nhận được\n")
        f.write("   b) Signed vs unsigned comparison trong MaxPool\n")
        f.write("      → luôn dùng signed int8 comparison\n")
        f.write("   c) Overflow trong GAP accumulator\n")
        f.write("      → cần ít nhất 24-bit signed accumulator (128×16×16=32768×127)\n")
        f.write("   d) Weight layout DW: [1, KH, KW, Cout] không phải [Cout, KH, KW, 1]\n")
        f.write("   e) ZP_weight = 0 cho int8 per-channel (KHÔNG phải 128)\n")
        f.write("   f) ReLU so sánh với out_zp (không phải 0 tuyệt đối)\n")
        f.write("      → max(out_val, zp_out)\n")
        f.write("   g) use_bias=False cho TẤT CẢ Conv layers (DW + PW)\n")
        f.write("      → bias tensor của conv = zeros, KHÔNG đóng góp vào acc\n")
        f.write("      → Chỉ FC1 và FC2 mới có bias thực sự (use_bias=True)\n\n")

        f.write("8. DEBUG FLOW:\n")
        f.write("   Nếu final output sai → binary search từng layer:\n")
        f.write("   1. Check layer_00_INPUT khớp không\n")
        f.write("   2. Check layer đầu tiên (DW Block1)\n")
        f.write("   3. Tiếp tục theo pipeline cho đến khi tìm layer bị sai\n")
        f.write("   4. Trong layer sai: check từng phần (MAC, requant, ReLU)\n")
        f.write("   5. Xem INTEGER_ONLY_DEMO trong layer_trace.py output để trace\n\n")

        f.write("9. QUICK START COMMAND:\n")
        f.write("   # Chạy inference + sinh ra tất cả golden files:\n")
        f.write("   python algo_verify.py --model drowsiness_asic_int8.tflite \\\n")
        f.write("                         --image face.jpg --out verify_output/\n\n")
        f.write("   # Hoặc từ pixel dump (Module 2):\n")
        f.write("   python algo_verify.py --model drowsiness_asic_int8.tflite \\\n")
        f.write("                         --pixels pixel_dumps/XXX_pixels_hex.txt \\\n")
        f.write("                         --out verify_output/\n")

    print(f"  💾 {p}")
    return p


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN ORCHESTRATOR — MAP TENSORS → LAYERS
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: NUMPY SIMULATION cho intermediate layers
# (XNNPACK không expose conv/pool activations → phải simulate lại bằng float32)
# ═════════════════════════════════════════════════════════════════════════════

def make_tensor_info(name, data_int8, scale, zp, scales_arr=None):
    """Tạo tensor info dict từ numpy array"""
    arr = np.array(data_int8, dtype=np.int8)
    if scales_arr is None:
        scales_arr = np.array([scale], dtype=np.float32)
    return {
        'name': name, 'shape': list(arr.shape), 'dtype': 'int8',
        'data': arr, 'scale': scale, 'zp': zp,
        'scales_arr': scales_arr, 'zps_arr': np.array([zp]),
    }


def sim_depthwise_conv(inp_t, dw_kernel, dw_bias, out_scale, out_zp,
                       stride=1, padding='SAME', has_relu=False):
    """
    Simulate DepthwiseConv2D bằng numpy float32.
    Input  : int8 [1, H, W, C]
    Kernel : int8 [1, KH, KW, C]  (TFLite DW layout)
    Bias   : int32 [C]
    Output : int8 [1, OH, OW, C]
    """
    inp_data   = dw_kernel['data'] if isinstance(dw_kernel, dict) else dw_kernel
    # Dùng tên rõ ràng
    x_data     = inp_t['data'].astype(np.float32)    # [1,H,W,C]
    x_sc       = inp_t['scale']
    x_zp       = inp_t['zp']
    w_data     = dw_kernel['data'].astype(np.float32) # [1,KH,KW,C]
    w_scales   = dw_kernel['scales_arr']
    b_data     = dw_bias['data'].astype(np.float64) if dw_bias is not None else None

    _, H, W, C  = x_data.shape
    _, KH, KW, Cout = w_data.shape   # Cout == C for depthwise

    if padding.upper() == 'SAME':
        pad_h = max(KH - 1, 0) // 2
        pad_w = max(KW - 1, 0) // 2
        OH = int(np.ceil(H / stride))
        OW = int(np.ceil(W / stride))
    else:
        pad_h = pad_w = 0
        OH = (H - KH) // stride + 1
        OW = (W - KW) // stride + 1

    # Pad input
    x_pad = np.pad(x_data, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)),
                   mode='constant', constant_values=x_zp)

    out = np.zeros((1, OH, OW, C), dtype=np.float64)
    for ch in range(C):
        sw = float(w_scales[ch % len(w_scales)])
        for oh in range(OH):
            for ow in range(OW):
                acc = 0.0
                for kh in range(KH):
                    for kw in range(KW):
                        ih = oh * stride + kh
                        iw = ow * stride + kw
                        x_val = float(x_pad[0, ih, iw, ch]) - x_zp
                        w_val = float(w_data[0, kh, kw, ch])
                        acc  += x_val * w_val
                if b_data is not None:
                    acc += float(b_data[ch % len(b_data)])
                # Dequant: acc → float
                real_val = x_sc * sw * acc
                # Requant → out_scale / out_zp
                q_val = real_val / out_scale + out_zp
                out[0, oh, ow, ch] = q_val

    out_int8 = np.clip(np.round(out), -128, 127).astype(np.int8)
    if has_relu:
        out_int8 = np.maximum(out_int8, out_zp).astype(np.int8)
    return out_int8


def sim_pointwise_conv(inp_t, pw_kernel, pw_bias, out_scale, out_zp, has_relu=False):
    """
    Simulate PointwiseConv2D (1×1 Conv2D) bằng numpy.
    Kernel TFLite layout: [Cout, 1, 1, Cin]
    """
    x_data   = inp_t['data'].astype(np.float64)
    x_sc     = inp_t['scale']
    x_zp     = inp_t['zp']
    w_data   = pw_kernel['data'].astype(np.float64)  # [Cout,1,1,Cin]
    w_scales = pw_kernel['scales_arr']
    b_data   = pw_bias['data'].astype(np.float64) if pw_bias is not None else None

    _, H, W, Cin = x_data.shape
    Cout = w_data.shape[0]

    out = np.zeros((1, H, W, Cout), dtype=np.float64)
    for co in range(Cout):
        sw = float(w_scales[co % len(w_scales)])
        w_vec = w_data[co, 0, 0, :]   # [Cin]
        # Dot product với tất cả spatial positions cùng lúc
        x_adj = x_data[0] - x_zp       # [H, W, Cin]
        acc   = x_adj @ w_vec           # [H, W]
        if b_data is not None:
            acc += float(b_data[co % len(b_data)])
        real_val = x_sc * sw * acc
        out[0, :, :, co] = real_val / out_scale + out_zp

    out_int8 = np.clip(np.round(out), -128, 127).astype(np.int8)
    if has_relu:
        out_int8 = np.maximum(out_int8, out_zp).astype(np.int8)
    return out_int8


def sim_maxpool(inp_t, pool_size=2, stride=2):
    """MaxPool2D simulation — integer passthrough, không cần requant"""
    x = inp_t['data']   # [1, H, W, C]
    _, H, W, C = x.shape
    OH = H // stride
    OW = W // stride
    out = np.full((1, OH, OW, C), -128, dtype=np.int8)
    for oh in range(OH):
        for ow in range(OW):
            window = x[0, oh*stride:oh*stride+pool_size,
                          ow*stride:ow*stride+pool_size, :]  # [PS,PS,C]
            # Signed max
            out[0, oh, ow, :] = window.reshape(-1, C).max(axis=0)
    return out


def sim_gap(inp_t, out_scale, out_zp):
    """GlobalAveragePooling2D simulation"""
    x = inp_t['data'].astype(np.float64)  # [1,H,W,C]
    x_sc = inp_t['scale']
    x_zp = inp_t['zp']
    _, H, W, C = x.shape
    x_adj = x[0] - x_zp                   # [H,W,C] float
    avg   = x_adj.sum(axis=(0,1)) / (H*W) # [C]
    real_val = x_sc * avg
    q_val    = real_val / out_scale + out_zp
    out_int8 = np.clip(np.round(q_val), -128, 127).astype(np.int8)
    return out_int8.reshape(1, C)


def sim_dense(inp_t, w_tensor, b_tensor, out_scale, out_zp, has_relu=False):
    """
    Dense (FC) layer simulation.
    Weight layout: [Cout, Cin] (2D) hoặc [Cout, 1, 1, Cin] (4D từ TFLite).
    """
    x_data   = inp_t['data'].astype(np.float64).flatten()   # [Cin]
    x_sc     = inp_t['scale']
    x_zp     = inp_t['zp']
    w_data   = w_tensor['data'].astype(np.float64)
    w_scales = w_tensor['scales_arr']
    b_data   = b_tensor['data'].astype(np.float64) if b_tensor is not None else None

    # Flatten weight nếu > 2D
    if w_data.ndim == 4:
        w_data = w_data.reshape(w_data.shape[0], -1)   # [Cout, Cin]
    Cout, Cin = w_data.shape

    x_adj = x_data - x_zp   # [Cin]
    out   = np.zeros(Cout, dtype=np.float64)
    for co in range(Cout):
        sw  = float(w_scales[co % len(w_scales)])
        acc = float(x_adj @ w_data[co])
        if b_data is not None:
            acc += float(b_data[co % len(b_data)])
        real_val  = x_sc * sw * acc
        out[co]   = real_val / out_scale + out_zp

    out_int8 = np.clip(np.round(out), -128, 127).astype(np.int8)
    if has_relu:
        out_int8 = np.maximum(out_int8, out_zp).astype(np.int8)
    return out_int8.reshape(1, -1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: HARDCODED TENSOR MAPPING (từ debug_tensors.py)
# ═════════════════════════════════════════════════════════════════════════════
#
# Kiến trúc: SeparableConv2D với use_bias=False cho TẤT CẢ conv layers.
# Chỉ FC1 (Dense 32) và FC2 (Dense 1) mới có bias thực sự.
# Bias tensors của conv layers (idx 3,6,10,13,17,20,24,27) sẽ là zero vector.
#
# Kết quả debug cho thấy cấu trúc thực tế:
#
#  WEIGHT TENSORS (accessible):
#   idx=2  → DW  Block1 kernel [1,3,3,1]      (n_scales=1, bị misclassify)
#   idx=5  → PW  Block1 kernel [32,1,1,1]
#   idx=9  → DW  Block2 kernel [1,3,3,32]
#   idx=12 → PW  Block2 kernel [64,1,1,32]
#   idx=16 → DW  Block3 kernel [1,3,3,64]
#   idx=19 → PW  Block3 kernel [128,1,1,64]
#   idx=23 → DW  Block4 kernel [1,3,3,128]
#   idx=26 → PW  Block4 kernel [128,1,1,128]
#   idx=30 → FC1 weight        [32,128]
#   idx=33 → FC2 weight        [1,32]         (n_scales=1, bị misclassify)
#
#  BIAS TENSORS:
#   idx=3  [1]   → DW Block1 bias  ← ZEROS (use_bias=False)
#   idx=6  [32]  → PW Block1 bias  ← ZEROS (use_bias=False)
#   idx=10 [32]  → DW Block2 bias  ← ZEROS (use_bias=False)
#   idx=13 [64]  → PW Block2 bias  ← ZEROS (use_bias=False)
#   idx=17 [64]  → DW Block3 bias  ← ZEROS (use_bias=False)
#   idx=20 [128] → PW Block3 bias  ← ZEROS (use_bias=False)
#   idx=24 [128] → DW Block4 bias  ← ZEROS (use_bias=False)
#   idx=27 [128] → PW Block4 bias  ← ZEROS (use_bias=False)
#   idx=31 [32]  → FC1 bias        ← có giá trị thực (use_bias=True)
#   idx=34 [1]   → FC2 bias        ← có giá trị thực (use_bias=True)
#
#  ACTIVATION TENSORS accessible (TFLite + XNNPACK):
#   idx=0  [1,128,128,1] → INPUT
#   idx=29 [1,128]       → GAP output       ← GOLDEN (exact)
#   idx=32 [1,32]        → FC1+ReLU output  ← GOLDEN (exact)
#   idx=35 [1,1]         → Final output     ← GOLDEN (exact)
#
#  NOTE: Conv và Pool layer outputs (idx 4,7,8,11,14,15,18,21,22,25,28)
#        đều "inaccessible" → simulate bằng float32 numpy.

TENSOR_MAP = {
    # DW kernels:   [1, KH, KW, C]
    'dw1_kernel': {'idx': 2,  'layout': 'DW [1,KH,KW,1]'},
    'dw2_kernel': {'idx': 9,  'layout': 'DW [1,KH,KW,32]'},
    'dw3_kernel': {'idx': 16, 'layout': 'DW [1,KH,KW,64]'},
    'dw4_kernel': {'idx': 23, 'layout': 'DW [1,KH,KW,128]'},
    # PW kernels:   [Cout, 1, 1, Cin]
    'pw1_kernel': {'idx': 5,  'layout': 'PW [32,1,1,1]'},
    'pw2_kernel': {'idx': 12, 'layout': 'PW [64,1,1,32]'},
    'pw3_kernel': {'idx': 19, 'layout': 'PW [128,1,1,64]'},
    'pw4_kernel': {'idx': 26, 'layout': 'PW [128,1,1,128]'},
    # FC weights:   [Cout, Cin]
    'fc1_weight': {'idx': 30, 'layout': 'FC [32,128]'},
    'fc2_weight': {'idx': 33, 'layout': 'FC [1,32]'},
    # Biases: int32
    'dw1_bias': {'idx': 3},    # [1]
    'pw1_bias': {'idx': 6},    # [32]
    'dw2_bias': {'idx': 10},   # [32]
    'pw2_bias': {'idx': 13},   # [64]
    'dw3_bias': {'idx': 17},   # [64]
    'pw3_bias': {'idx': 20},   # [128]
    'dw4_bias': {'idx': 24},   # [128]
    'pw4_bias': {'idx': 27},   # [128]
    'fc1_bias': {'idx': 31},   # [32]
    'fc2_bias': {'idx': 34},   # [1]
    # Activations accessible:
    'gap_out': {'idx': 29},    # [1,128]   GOLDEN
    'fc1_out': {'idx': 32},    # [1,32]    GOLDEN
    'fc2_out': {'idx': 33},    # [1,32]    — actually weight; output is idx=35
    'out':     {'idx': 35},    # [1,1]     GOLDEN
}


def get_t(interp, idx):
    """Lấy tensor từ interpreter theo index, trả về tensor info dict"""
    data = interp.get_tensor(idx)
    all_det = interp.get_tensor_details()
    t_det = next((t for t in all_det if t['index'] == idx), None)
    if t_det is None or data is None:
        return None
    qp = t_det['quantization_parameters']
    sc_arr = qp['scales'].copy()   if len(qp['scales'])      > 0 else np.array([1.0])
    zp_arr = qp['zero_points'].copy() if len(qp['zero_points']) > 0 else np.array([0])
    scale  = float(sc_arr[0])
    zp     = int(zp_arr[0])
    return {
        'name': t_det['name'], 'shape': list(data.shape),
        'dtype': str(data.dtype), 'data': data.copy(),
        'scale': scale, 'zp': zp,
        'scales_arr': sc_arr, 'zps_arr': zp_arr,
    }


def generate_all_layer_files(model_path, q07_input, out_dir):
    """
    Hàm chính: chạy inference, map tensors đúng theo TENSOR_MAP,
    simulate intermediate layers bằng numpy, sinh file cho mỗi layer.
    """
    print("\n⚙️  Running inference...")
    interp, inp_det, out_det, all_details, inp_int8, logit, prob = \
        run_inference_collect(model_path, q07_input)

    out_raw_int8 = int(interp.get_tensor(out_det['index']).flatten()[0])

    print("\n📂 Loading tensors theo hardcoded mapping...")

    # ── Load tất cả tensors cần thiết ────────────────────────────────────────
    def gt(key):
        return get_t(interp, TENSOR_MAP[key]['idx'])

    # Kernels + biases
    dw1_k = gt('dw1_kernel')
    dw2_k = gt('dw2_kernel')
    dw3_k = gt('dw3_kernel')
    dw4_k = gt('dw4_kernel')
    pw1_k = gt('pw1_kernel')
    pw2_k = gt('pw2_kernel')
    pw3_k = gt('pw3_kernel')
    pw4_k = gt('pw4_kernel')
    fc1_w = gt('fc1_weight')
    fc2_w = gt('fc2_weight')

    dw1_b = gt('dw1_bias')
    pw1_b = gt('pw1_bias')
    dw2_b = gt('dw2_bias')
    pw2_b = gt('pw2_bias')
    dw3_b = gt('dw3_bias')
    pw3_b = gt('pw3_bias')
    dw4_b = gt('dw4_bias')
    pw4_b = gt('pw4_bias')
    fc1_b = gt('fc1_bias')
    fc2_b = gt('fc2_bias')

    # Accessible activations (GOLDEN)
    gap_out_t = gt('gap_out')   # idx=29 [1,128]
    fc1_out_t = gt('fc1_out')   # idx=32 [1,32]  (FC1+ReLU fused)
    out_t     = get_t(interp, out_det['index'])  # idx=35 [1,1]

    print(f"  ✅ Kernels + biases loaded")
    print(f"  ✅ Accessible activations: GAP, FC1+ReLU, Final Output")
    print(f"  ℹ️  Conv/Pool outputs sẽ được simulate bằng float32 numpy")

    # ── Tìm scales của intermediate activations từ tensor details ─────────────
    # Các tensors inaccessible nhưng vẫn có scale/zp trong tensor_details
    # Dùng để set đúng scale khi simulate
    scale_map = {}   # idx → (scale, zp)
    for t in all_details:
        qp = t['quantization_parameters']
        if len(qp['scales']) == 1:
            scale_map[t['index']] = (float(qp['scales'][0]),
                                     int(qp['zero_points'][0]))

    def get_scale(idx, default_scale=0.01, default_zp=0):
        return scale_map.get(idx, (default_scale, default_zp))

    # Scale của các intermediate activation tensors
    # (lấy từ tensor details ngay cả khi inaccessible)
    dw1_out_sc, dw1_out_zp = get_scale(4)   # DW Block1 out
    pw1_out_sc, pw1_out_zp = get_scale(7)   # PW Block1+ReLU out
    pool1_sc,   pool1_zp   = get_scale(8)   # Pool1 out
    dw2_out_sc, dw2_out_zp = get_scale(11)  # DW Block2 out
    pw2_out_sc, pw2_out_zp = get_scale(14)  # PW Block2+ReLU out
    pool2_sc,   pool2_zp   = get_scale(15)  # Pool2 out
    dw3_out_sc, dw3_out_zp = get_scale(18)  # DW Block3 out
    pw3_out_sc, pw3_out_zp = get_scale(21)  # PW Block3+ReLU out
    pool3_sc,   pool3_zp   = get_scale(22)  # Pool3 out
    dw4_out_sc, dw4_out_zp = get_scale(25)  # DW Block4 out
    pw4_out_sc, pw4_out_zp = get_scale(28)  # PW Block4+ReLU out

    print("\n💾 Generating layer files...")
    os.makedirs(out_dir, exist_ok=True)
    layer_dirs = []
    layer_num  = 0

    # ── INPUT tensor info ────────────────────────────────────────────────────
    inp_info = make_tensor_info(
        'INPUT', inp_int8,
        float(inp_det['quantization_parameters']['scales'][0]),
        int(inp_det['quantization_parameters']['zero_points'][0]))

    d = write_algo_input(out_dir, inp_int8, inp_det)
    layer_dirs.append(d); layer_num += 1

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK 1: DW(1) → PW(32)+ReLU → MaxPool → [64×64×32]
    # ─────────────────────────────────────────────────────────────────────────
    print("  📐 Simulating Block 1...")

    # DW Block1: input=[1,128,128,1], kernel=[1,3,3,1]
    dw1_out_data = sim_depthwise_conv(inp_info, dw1_k, dw1_b,
                                      dw1_out_sc, dw1_out_zp, stride=1,
                                      padding='SAME', has_relu=False)
    dw1_out_info = make_tensor_info('dw1_out', dw1_out_data,
                                    dw1_out_sc, dw1_out_zp,
                                    np.array([dw1_out_sc]))
    d = write_algo_depthwise_conv(out_dir, layer_num, 'conv1', inp_info,
                                  dw1_k, dw1_b, dw1_out_info,
                                  stride=1, padding_mode='SAME')
    layer_dirs.append(d); layer_num += 1

    # PW Block1 + ReLU: input=[1,128,128,1], kernel=[32,1,1,1]
    pw1_out_data = sim_pointwise_conv(dw1_out_info, pw1_k, pw1_b,
                                      pw1_out_sc, pw1_out_zp, has_relu=True)
    pw1_out_info = make_tensor_info('pw1_out', pw1_out_data,
                                    pw1_out_sc, pw1_out_zp,
                                    np.array([pw1_out_sc]))
    d = write_algo_pointwise_conv(out_dir, layer_num, 'conv1', dw1_out_info,
                                   pw1_k, pw1_b, pw1_out_info, has_relu=True)
    layer_dirs.append(d); layer_num += 1

    # MaxPool1: [1,128,128,32] → [1,64,64,32]
    pool1_data = sim_maxpool(pw1_out_info, pool_size=2, stride=2)
    pool1_info = make_tensor_info('pool1_out', pool1_data,
                                  pw1_out_sc, pw1_out_zp,
                                  np.array([pw1_out_sc]))
    d = write_algo_maxpool(out_dir, layer_num, 'pool1', pw1_out_info,
                            pool1_info, pool_size=2, stride=2)
    layer_dirs.append(d); layer_num += 1

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK 2: DW(32) → PW(64)+ReLU → MaxPool → [32×32×64]
    # ─────────────────────────────────────────────────────────────────────────
    print("  📐 Simulating Block 2...")

    dw2_out_data = sim_depthwise_conv(pool1_info, dw2_k, dw2_b,
                                      dw2_out_sc, dw2_out_zp, stride=1,
                                      padding='SAME', has_relu=False)
    dw2_out_info = make_tensor_info('dw2_out', dw2_out_data,
                                    dw2_out_sc, dw2_out_zp,
                                    np.array([dw2_out_sc]))
    d = write_algo_depthwise_conv(out_dir, layer_num, 'conv2', pool1_info,
                                  dw2_k, dw2_b, dw2_out_info,
                                  stride=1, padding_mode='SAME')
    layer_dirs.append(d); layer_num += 1

    pw2_out_data = sim_pointwise_conv(dw2_out_info, pw2_k, pw2_b,
                                      pw2_out_sc, pw2_out_zp, has_relu=True)
    pw2_out_info = make_tensor_info('pw2_out', pw2_out_data,
                                    pw2_out_sc, pw2_out_zp,
                                    np.array([pw2_out_sc]))
    d = write_algo_pointwise_conv(out_dir, layer_num, 'conv2', dw2_out_info,
                                   pw2_k, pw2_b, pw2_out_info, has_relu=True)
    layer_dirs.append(d); layer_num += 1

    pool2_data = sim_maxpool(pw2_out_info, pool_size=2, stride=2)
    pool2_info = make_tensor_info('pool2_out', pool2_data,
                                  pw2_out_sc, pw2_out_zp,
                                  np.array([pw2_out_sc]))
    d = write_algo_maxpool(out_dir, layer_num, 'pool2', pw2_out_info,
                            pool2_info, pool_size=2, stride=2)
    layer_dirs.append(d); layer_num += 1

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK 3: DW(64) → PW(128)+ReLU → MaxPool → [16×16×128]
    # ─────────────────────────────────────────────────────────────────────────
    print("  📐 Simulating Block 3...")

    dw3_out_data = sim_depthwise_conv(pool2_info, dw3_k, dw3_b,
                                      dw3_out_sc, dw3_out_zp, stride=1,
                                      padding='SAME', has_relu=False)
    dw3_out_info = make_tensor_info('dw3_out', dw3_out_data,
                                    dw3_out_sc, dw3_out_zp,
                                    np.array([dw3_out_sc]))
    d = write_algo_depthwise_conv(out_dir, layer_num, 'conv3', pool2_info,
                                  dw3_k, dw3_b, dw3_out_info,
                                  stride=1, padding_mode='SAME')
    layer_dirs.append(d); layer_num += 1

    pw3_out_data = sim_pointwise_conv(dw3_out_info, pw3_k, pw3_b,
                                      pw3_out_sc, pw3_out_zp, has_relu=True)
    pw3_out_info = make_tensor_info('pw3_out', pw3_out_data,
                                    pw3_out_sc, pw3_out_zp,
                                    np.array([pw3_out_sc]))
    d = write_algo_pointwise_conv(out_dir, layer_num, 'conv3', dw3_out_info,
                                   pw3_k, pw3_b, pw3_out_info, has_relu=True)
    layer_dirs.append(d); layer_num += 1

    pool3_data = sim_maxpool(pw3_out_info, pool_size=2, stride=2)
    pool3_info = make_tensor_info('pool3_out', pool3_data,
                                  pw3_out_sc, pw3_out_zp,
                                  np.array([pw3_out_sc]))
    d = write_algo_maxpool(out_dir, layer_num, 'pool3', pw3_out_info,
                            pool3_info, pool_size=2, stride=2)
    layer_dirs.append(d); layer_num += 1

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK 4: DW(128) → PW(128)+ReLU (không có MaxPool)
    # ─────────────────────────────────────────────────────────────────────────
    print("  📐 Simulating Block 4...")

    dw4_out_data = sim_depthwise_conv(pool3_info, dw4_k, dw4_b,
                                      dw4_out_sc, dw4_out_zp, stride=1,
                                      padding='SAME', has_relu=False)
    dw4_out_info = make_tensor_info('dw4_out', dw4_out_data,
                                    dw4_out_sc, dw4_out_zp,
                                    np.array([dw4_out_sc]))
    d = write_algo_depthwise_conv(out_dir, layer_num, 'conv4', pool3_info,
                                  dw4_k, dw4_b, dw4_out_info,
                                  stride=1, padding_mode='SAME')
    layer_dirs.append(d); layer_num += 1

    pw4_out_data = sim_pointwise_conv(dw4_out_info, pw4_k, pw4_b,
                                      pw4_out_sc, pw4_out_zp, has_relu=True)
    pw4_out_info = make_tensor_info('pw4_out', pw4_out_data,
                                    pw4_out_sc, pw4_out_zp,
                                    np.array([pw4_out_sc]))
    d = write_algo_pointwise_conv(out_dir, layer_num, 'conv4', dw4_out_info,
                                   pw4_k, pw4_b, pw4_out_info, has_relu=True)
    layer_dirs.append(d); layer_num += 1

    # ─────────────────────────────────────────────────────────────────────────
    # GAP — output ACCESSIBLE (idx=29), dùng làm GOLDEN
    # ─────────────────────────────────────────────────────────────────────────
    print("  📐 GAP (GOLDEN from TFLite)...")
    # Dùng simulated PW4 output làm input
    # Output dùng GOLDEN từ TFLite (idx=29)
    d = write_algo_gap(out_dir, layer_num, 'gap', pw4_out_info, gap_out_t)
    layer_dirs.append(d); layer_num += 1

    # ─────────────────────────────────────────────────────────────────────────
    # FC1 + ReLU — output ACCESSIBLE (idx=32), dùng làm GOLDEN
    # ─────────────────────────────────────────────────────────────────────────
    print("  📐 FC1+ReLU (GOLDEN from TFLite)...")
    d = write_algo_dense(out_dir, layer_num, 'fc1',
                         gap_out_t, fc1_w, fc1_b, fc1_out_t, has_relu=True)
    layer_dirs.append(d); layer_num += 1

    # ─────────────────────────────────────────────────────────────────────────
    # FC2 (output logit) — output ACCESSIBLE (idx=35), dùng làm GOLDEN
    # ─────────────────────────────────────────────────────────────────────────
    print("  📐 FC2/Output (GOLDEN from TFLite)...")
    d = write_algo_dense(out_dir, layer_num, 'output',
                         fc1_out_t, fc2_w, fc2_b, out_t, has_relu=False)
    layer_dirs.append(d); layer_num += 1

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print(f"\n📊 Pipeline overview...")
    write_pipeline_overview(out_dir, model_path, logit, prob,
                            out_raw_int8, inp_det, out_det, layer_dirs)

    print(f"\n📖 Verification guide...")
    write_verification_guide(out_dir, logit, prob, out_raw_int8, out_det)

    return logit, prob, out_raw_int8, layer_dirs


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MODULE 5 — Sinh algorithm description + golden reference files "
                    "cho từng layer để verify RTL/ASIC implementation")
    parser.add_argument("--model",  default="drowsiness_asic_int8.tflite",
                        help="Path đến TFLite model")
    parser.add_argument("--pixels", default=None,
                        help="File hex từ Module 2 (capture_pixel_dump.py)")
    parser.add_argument("--image",  default=None,
                        help="Ảnh đầu vào (thay thế cho --pixels)")
    parser.add_argument("--out",    default="algo_verify_output",
                        help="Thư mục output")
    args = parser.parse_args()

    print("=" * 72)
    print("MODULE 5 — ALGORITHM VERIFICATION REFERENCE GENERATOR")
    print("=" * 72)
    print(f"Model : {args.model}")
    print(f"Output: {args.out}/")

    if not os.path.exists(args.model):
        print(f"\n❌ Không tìm thấy model: {args.model}")
        sys.exit(1)

    if not args.pixels and not args.image:
        print("\n❌ Cần chỉ định --pixels hoặc --image")
        parser.print_help()
        sys.exit(1)

    print("\n📂 Loading input...")
    if args.pixels:
        q07 = load_from_hex_file(args.pixels)
    else:
        q07 = load_from_image(args.image)

    print(f"  Q0.7 input: shape={q07.shape}, range=[{q07.min()}, {q07.max()}]")

    logit, prob, out_raw, layer_dirs = generate_all_layer_files(
        args.model, q07, args.out)

    label = "ACTIVE 😊" if prob >= ACTIVE_THRESHOLD else "FATIGUE 😴"
    conf  = prob * 100 if prob >= ACTIVE_THRESHOLD else (1 - prob) * 100

    print(f"\n{'=' * 72}")
    print(f"✅ HOÀN THÀNH — {len(layer_dirs)} layer folders sinh ra")
    print(f"{'=' * 72}")
    print(f"\n🎯 PREDICTION: {label}  ({conf:.1f}% confidence)")
    print(f"   logit        = {logit:.6f}")
    print(f"   prob_active  = {prob:.6f}")
    print(f"   out_raw_int8 = {int(out_raw)}")
    print(f"\n📁 Output: {args.out}/")
    print(f"\n📌 Workflow cho RTL engineer:")
    print(f"  1. Đọc PIPELINE_OVERVIEW.txt  — sơ đồ toàn pipeline")
    print(f"  2. Đọc VERIFICATION_GUIDE.txt — hướng dẫn verification")
    print(f"  3. Với từng layer folder:")
    print(f"     a. Đọc ALGO.txt     — implement đúng thuật toán")
    print(f"     b. Load input + kernel + bias từ *.hex.txt / *.int.txt")
    print(f"     c. So sánh RTL output với output.int.txt")
    print(f"     d. Tolerance: ±1 LSB")
    print(f"\n⚠️  ACCURACY NOTE:")
    print(f"   • Conv/Pool output files = float32 simulation (±1-2 LSB vs TFLite)")
    print(f"   • GAP, FC1, FC2, Output  = GOLDEN EXACT từ TFLite runtime")


if __name__ == "__main__":
    main()