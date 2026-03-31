"""
MODULE 3 — Đọc weights model AI → lưu ra file TXT
===================================================
FIX: Thêm dummy invoke() trước get_tensor() để tránh lỗi
     "Tensor data is null. Run allocate_tensors() first"

Chạy:
  python .\model\weight_dump.py --model .\model\drowsiness_asic_int8.tflite
  python .\model\weight_dump.py --model .\model\drowsiness_asic_int8.tflite --out my_weights
"""

import numpy as np
import argparse
import os
import json

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def load_model_tensors(model_path):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    print(f"\n📥 Input  : shape={inp_det['shape'].tolist()}, dtype={inp_det['dtype']}")
    print(f"   scale={inp_det['quantization_parameters']['scales']}")
    print(f"   zp   ={inp_det['quantization_parameters']['zero_points']}")
    print(f"📤 Output : shape={out_det['shape'].tolist()}, dtype={out_det['dtype']}")
    print(f"   scale={out_det['quantization_parameters']['scales']}")
    print(f"   zp   ={out_det['quantization_parameters']['zero_points']}")

    # ── FIX: chạy dummy invoke để map tensor memory ───────────────────────
    # Weight tensors trong TFLite chỉ accessible qua get_tensor() SAU invoke().
    print("\n⚙️  Running dummy invoke để unlock weight tensors...")
    dummy = np.zeros(inp_det['shape'], dtype=np.int8)
    interp.set_tensor(inp_det['index'], dummy)
    interp.invoke()
    print("   ✅ Done")

    all_details = interp.get_tensor_details()
    tensors = []

    for t in all_details:
        qp = t['quantization_parameters']
        if len(qp['scales']) == 0:
            continue

        try:
            data = interp.get_tensor(t['index'])
        except Exception as e:
            print(f"   ⚠️  Skip [{t['index']}] {t['name']}: {e}")
            continue

        if data is None or data.size == 0:
            continue

        if data.dtype == np.int8:
            kind = 'weight'
        elif data.dtype == np.int32:
            kind = 'bias'
        else:
            kind = 'activation'

        tensors.append({
            'index':       t['index'],
            'name':        t['name'],
            'shape':       list(data.shape),
            'dtype':       str(data.dtype),
            'data':        data.copy(),
            'scales':      qp['scales'].copy(),
            'zero_points': qp['zero_points'].copy(),
            'n_elements':  int(data.size),
            'size_bytes':  int(data.nbytes),
            'kind':        kind,
        })

    return tensors, inp_det, out_det


def safe_name(name):
    """Chuyển tensor name thành tên file an toàn"""
    for c in r'/\;:. ':
        name = name.replace(c, '_')
    return name.strip('_')


def dump_weight_tensor(t, out_dir):
    """
    Dump weight int8 tensor ra nhiều định dạng.

    NOTE về layout TFLite Conv2D:
      - TFLite lưu kernel theo thứ tự (Cout, H, W, Cin)
      - Mỗi output channel có 1 scale riêng (per-channel quantization)
    """
    data, scales, zps = t['data'], t['scales'], t['zero_points']
    sn   = safe_name(t['name'])
    flat = data.flatten()
    files = {}

    # ── Decimal ─────────────────────────────────────────────────────────
    p = os.path.join(out_dir, f"{sn}.dec.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"// {t['name']}\n")
        f.write(f"// Shape: {t['shape']}, int8, {len(scales)} ch scales\n\n")
        for v in flat:
            f.write(f"{int(v)}\n")
    files['dec'] = p

    # ── Hex ($readmemh) ─────────────────────────────────────────────────
    p = os.path.join(out_dir, f"{sn}.hex.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"// {t['name']}\n")
        f.write(f"// $readmemh(\"{os.path.basename(p)}\", mem);\n\n")
        for v in flat:
            f.write(f"{int(v) & 0xFF:02X}\n")
    files['hex'] = p

    # ── Scales per channel ───────────────────────────────────────────────
    p = os.path.join(out_dir, f"{sn}.scales.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"// Scales: {t['name']}\n")
        f.write(f"// real_value = scale[ch] * int8_value  (zero_point=0 untuk weight)\n\n")
        f.write(f"// ch,  scale_f32,             scale_Q8.24\n")
        for i, (s, z) in enumerate(zip(scales, zps)):
            q24 = int(round(float(s) * 2**24)) & 0xFFFFFFFF
            f.write(f"{i:4d},  {float(s):.12f},  {q24:08X}  // zp={int(z)}\n")
    files['scales'] = p

    # ── Shaped view (chỉ với tensor nhỏ) ────────────────────────────────
    if t['n_elements'] <= 2048 and len(t['shape']) in (2, 4):
        p = os.path.join(out_dir, f"{sn}.shaped.txt")
        with open(p, 'w', encoding='utf-8') as f:
            f.write(f"// {t['name']} — shaped view\n")
            f.write(f"// Shape: {t['shape']}\n")
            sh = t['shape']
            if len(sh) == 4:
                # TFLite Conv2D layout: (Cout, H, W, Cin)
                Co, H, W, Ci = sh
                f.write(f"// Layout: (Cout={Co}, H={H}, W={W}, Cin={Ci})\n\n")
                for co in range(Co):
                    f.write(f"// Cout={co}  scale={float(scales[co % len(scales)]):.8f}\n")
                    for ci in range(Ci):
                        for h in range(H):
                            row = ' '.join(f"{int(data[co, h, w, ci]):4d}"
                                           for w in range(W))
                            f.write(f"//  ci={ci} h={h}: {row}\n")
                    f.write("\n")
            else:
                # Dense: (Cout, Cin)
                Ro, Ri = sh
                for r in range(Ro):
                    row = ' '.join(f"{int(data[r, c]):4d}" for c in range(Ri))
                    f.write(f"// out[{r:3d}]: {row}\n")
        files['shaped'] = p

    return files


def dump_bias_tensor(t, out_dir):
    sn   = safe_name(t['name'])
    flat = t['data'].flatten()
    p    = os.path.join(out_dir, f"{sn}.bias.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"// Bias: {t['name']}\n")
        f.write(f"// Shape: {t['shape']}, int32\n\n")
        f.write(f"// idx,  decimal,       hex_32bit\n")
        for i, v in enumerate(flat):
            iv = int(v)
            f.write(f"{i:5d},  {iv:12d},  {iv & 0xFFFFFFFF:08X}\n")
    return p


def compute_requant(S_in, weight_scales, S_out_layer):
    """
    Tính requantization params M = S_in * S_w / S_out_layer
    cho từng output channel.

    Args:
        S_in         : float — input tensor scale
        weight_scales: array — per-channel weight scales
        S_out_layer  : float — output scale của LAYER NÀY (không phải model output)
    """
    results = []
    for sw in weight_scales:
        M       = float(S_in) * float(sw) / float(S_out_layer)
        shift   = 0
        M_norm  = M
        while M_norm < 0.5 and shift < 31:
            M_norm *= 2
            shift  += 1
        M_int32 = int(round(M_norm * 2**30))
        results.append({
            'M_float':    M,
            'M_int32':    M_int32,
            'total_shift': shift + 30,
        })
    return results


def dump_requant_params(tensors, inp_det, out_det, out_dir):
    """
    Dump requantization parameters.

    NOTE: S_out được lấy từ scale của tensor activation SAU layer (nếu có).
    Ở đây dùng model output scale làm fallback; RTL engineer cần đối chiếu
    với scale của intermediate activation tensor tương ứng.
    """
    S_in   = float(inp_det['quantization_parameters']['scales'][0])
    S_out  = float(out_det['quantization_parameters']['scales'][0])
    zp_out = int(out_det['quantization_parameters']['zero_points'][0])

    p = os.path.join(out_dir, "REQUANT_PARAMS.txt")
    with open(p, 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write("REQUANTIZATION PARAMETERS CHO ASIC RTL\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"S_input        = {S_in:.12f}\n")
        f.write(f"S_output_model = {S_out:.12f}  "
                f"(scale của model output — dùng làm fallback)\n")
        f.write(f"ZP_out         = {zp_out}\n\n")
        f.write("⚠️  LƯU Ý: S_out ở đây là scale của model output CUỐI CÙNG.\n")
        f.write("   Trong thực tế mỗi layer có S_out riêng (scale activation tiếp theo).\n")
        f.write("   Hãy dùng layer_trace.py để lấy đúng scale từng intermediate tensor.\n\n")
        f.write("Datapath:\n")
        f.write("  int32  acc    = MAC(int8_x, int8_w) + int32_bias\n")
        f.write("  int64  scaled = (int64)acc * (int64)M_int32\n")
        f.write("  int32  shifted= scaled >> total_shift\n")
        f.write("  int8   out    = clip(shifted + zp_out, -128, 127)\n\n")

        weight_tensors = [t for t in tensors if t['kind'] == 'weight']
        for t in weight_tensors:
            params = compute_requant(S_in, t['scales'], S_out)
            f.write(f"\n{'─' * 72}\n")
            f.write(f"Layer: {t['name']}\nShape: {t['shape']}\n")
            f.write(f"{'Ch':>5}  {'M_float':>18}  {'M_int32':>12}  {'shift':>6}\n")
            f.write(f"{'─' * 72}\n")
            for ch, p_ in enumerate(params):
                f.write(f"{ch:>5}  {p_['M_float']:>18.12f}  "
                        f"{p_['M_int32']:>12d}  {p_['total_shift']:>6d}\n")

    print(f"  💾 {p}  ← QUAN TRỌNG")
    return p


def dump_model_summary(tensors, inp_det, out_det, out_dir):
    p  = os.path.join(out_dir, "MODEL_SUMMARY.txt")
    tw = tb = 0
    with open(p, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("MODEL WEIGHT SUMMARY\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Idx':>4}  {'Name':<50}  {'Shape':<20}  "
                f"{'Dtype':<11}  {'N':>9}  {'KB':>6}  Kind\n")
        f.write("─" * 110 + "\n")
        for t in tensors:
            if t['kind'] not in ('weight', 'bias'):
                continue
            kb = t['size_bytes'] / 1024
            f.write(f"{t['index']:>4}  {t['name']:<50}  {str(t['shape']):<20}  "
                    f"{t['dtype']:<11}  {t['n_elements']:>9,}  {kb:>6.2f}  {t['kind']}\n")
            if   t['kind'] == 'weight': tw += t['n_elements']
            elif t['kind'] == 'bias':   tb += t['n_elements']
        f.write("─" * 110 + "\n")
        f.write(f"\nWeight SRAM : {tw:,} bytes  = {tw / 1024:.2f} KB  (int8)\n")
        f.write(f"Bias ROM    : {tb * 4:,} bytes  = {tb * 4 / 1024:.2f} KB  (int32)\n")
        S_in   = float(inp_det['quantization_parameters']['scales'][0])
        ZP_in  = int(inp_det['quantization_parameters']['zero_points'][0])
        S_out  = float(out_det['quantization_parameters']['scales'][0])
        ZP_out = int(out_det['quantization_parameters']['zero_points'][0])
        f.write(f"\nInput  scale={S_in:.10f},  zp={ZP_in}\n")
        f.write(f"Output scale={S_out:.10f},  zp={ZP_out}\n")
    print(f"  💾 {p}")
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Dump INT8 weights từ TFLite model ra file txt/hex cho RTL")
    parser.add_argument("--model", default="drowsiness_asic_int8.tflite")
    parser.add_argument("--out",   default="weight_dumps")
    args = parser.parse_args()

    print("=" * 70)
    print("MODULE 3 — WEIGHT DUMP")
    print("=" * 70)
    print(f"Model : {args.model}")
    print(f"Output: {args.out}/")

    if not os.path.exists(args.model):
        print(f"\n❌ Không tìm thấy: {args.model}")
        return

    os.makedirs(args.out, exist_ok=True)

    print("\n📂 Loading tensors...")
    tensors, inp_det, out_det = load_model_tensors(args.model)

    weights = [t for t in tensors if t['kind'] == 'weight']
    biases  = [t for t in tensors if t['kind'] == 'bias']

    print(f"\n✅ Found: {len(weights)} weight (int8),  {len(biases)} bias (int32)")

    print(f"\n💾 Dumping...")
    for t in tensors:
        if t['kind'] not in ('weight', 'bias'):
            continue
        tag = "W" if t['kind'] == 'weight' else "B"
        print(f"  [{t['index']:3d}|{tag}] {t['name']:<55} shape={t['shape']}")
        if t['kind'] == 'weight':
            files = dump_weight_tensor(t, args.out)
            print(f"         → {list(files.keys())}")
        else:
            bp = dump_bias_tensor(t, args.out)
            print(f"         → {os.path.basename(bp)}")

    print(f"\n📐 Requant params...")
    dump_requant_params(tensors, inp_det, out_det, args.out)

    print(f"\n📋 Summary...")
    dump_model_summary(tensors, inp_det, out_det, args.out)

    # INDEX.json
    idx_path = os.path.join(args.out, "INDEX.json")
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model':        args.model,
            'input_scale':  float(inp_det['quantization_parameters']['scales'][0]),
            'input_zp':     int(inp_det['quantization_parameters']['zero_points'][0]),
            'output_scale': float(out_det['quantization_parameters']['scales'][0]),
            'output_zp':    int(out_det['quantization_parameters']['zero_points'][0]),
            'tensors': [
                {
                    'index':     t['index'],
                    'name':      t['name'],
                    'shape':     t['shape'],
                    'dtype':     t['dtype'],
                    'kind':      t['kind'],
                    'scale_min': float(t['scales'].min()),
                    'scale_max': float(t['scales'].max()),
                }
                for t in tensors if t['kind'] in ('weight', 'bias')
            ]
        }, f, indent=2)
    print(f"  💾 {idx_path}")

    tw = sum(t['n_elements'] for t in weights)
    tb = sum(t['n_elements'] for t in biases)
    print(f"\n{'=' * 70}")
    print(f"✅ DONE — {args.out}/")
    print(f"   Weight SRAM: {tw / 1024:.1f} KB  |  Bias ROM: {tb * 4 / 1024:.1f} KB")
    print(f"\n📌 Quan trọng nhất:")
    print(f"   REQUANT_PARAMS.txt  — M_int32 + shift cho RTL")
    print(f"   *.hex.txt           — dùng $readmemh() trong Verilog")
    print(f"   *.bias.txt          — int32 biases")


if __name__ == "__main__":
    main()