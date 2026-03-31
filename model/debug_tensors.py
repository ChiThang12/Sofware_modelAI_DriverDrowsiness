"""
DEBUG TOOL — Kiểm tra chính xác tensors TFLite expose
Chạy TRƯỚC khi chạy algo_verify.py để hiểu cấu trúc thực tế của model.

python .\model\debug_tensors.py --model model/drowsiness_asic_int8.tflite --image model/your_photo.jpg
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

IMG_SIZE     = 128
SCALE_FACTOR = 128.0


def load_from_image(image_path):
    import cv2
    frame = cv2.imread(image_path)
    fc    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.3, 5)
    if len(faces):
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
    else:
        roi = gray
    resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return np.clip((resized.astype(np.int16) >> 1), 0, 127).astype(np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    # Invoke với input thực
    if args.image:
        q07 = load_from_image(args.image)
    else:
        q07 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int8)

    inp_s  = float(inp_det['quantization_parameters']['scales'][0])
    inp_zp = int(inp_det['quantization_parameters']['zero_points'][0])
    fp32   = q07.astype(np.float32) / SCALE_FACTOR
    fp32   = fp32[np.newaxis, :, :, np.newaxis]
    inp_i8 = np.round(fp32 / inp_s + inp_zp).clip(-128, 127).astype(np.int8)

    interp.set_tensor(inp_det['index'], inp_i8)
    interp.invoke()

    all_details = interp.get_tensor_details()

    print("=" * 100)
    print("TẤT CẢ TENSORS TRONG MODEL (sau invoke)")
    print("=" * 100)
    print(f"{'Idx':>4}  {'Dtype':<10}  {'n_scales':>8}  {'Shape':<28}  Name")
    print("─" * 100)

    weights, biases, activations, others = [], [], [], []

    for t in sorted(all_details, key=lambda x: x['index']):
        idx  = t['index']
        qp   = t['quantization_parameters']
        n_sc = len(qp['scales'])

        try:
            data = interp.get_tensor(idx)
            accessible = True
        except Exception:
            data = None
            accessible = False

        if data is None or not accessible:
            shape_str = "N/A (inaccessible)"
            dtype_str = str(t['dtype'])
        else:
            shape_str = str(list(data.shape))
            dtype_str = str(data.dtype)

        print(f"{idx:>4}  {dtype_str:<10}  {n_sc:>8}  {shape_str:<28}  {t['name']}")

        if data is not None:
            if data.dtype == np.int8 and n_sc > 1:
                weights.append((idx, t['name'], list(data.shape), n_sc, data))
            elif data.dtype == np.int32:
                biases.append((idx, t['name'], list(data.shape), data))
            elif data.dtype == np.int8 and n_sc <= 1:
                activations.append((idx, t['name'], list(data.shape),
                                    float(qp['scales'][0]) if n_sc == 1 else None,
                                    int(qp['zero_points'][0]) if n_sc == 1 else None,
                                    data))

    print("\n" + "=" * 100)
    print(f"WEIGHT TENSORS (int8, n_scales > 1) — {len(weights)} total")
    print("=" * 100)
    for idx, name, shape, n_sc, data in weights:
        print(f"  [{idx:3d}]  shape={str(shape):<25}  n_ch_scales={n_sc:<5}  {name}")

    print("\n" + "=" * 100)
    print(f"BIAS TENSORS (int32) — {len(biases)} total")
    print("=" * 100)
    for idx, name, shape, data in biases:
        print(f"  [{idx:3d}]  shape={str(shape):<25}  {name}")

    print("\n" + "=" * 100)
    print(f"ACTIVATION TENSORS (int8, n_scales ≤ 1) — {len(activations)} total")
    print("=" * 100)
    for idx, name, shape, scale, zp, data in activations:
        rng = f"[{int(data.min())}, {int(data.max())}]"
        print(f"  [{idx:3d}]  shape={str(shape):<28}  scale={scale}  zp={zp}  range={rng}  {name}")

    # In ra thứ tự các op (nếu có)
    print("\n" + "=" * 100)
    print("THÔNG TIN INPUT / OUTPUT")
    print("=" * 100)
    print(f"  Input  idx={inp_det['index']}  shape={inp_det['shape'].tolist()}  "
          f"scale={inp_s:.10f}  zp={inp_zp}")
    out_s  = float(out_det['quantization_parameters']['scales'][0])
    out_zp = int(out_det['quantization_parameters']['zero_points'][0])
    out_d  = interp.get_tensor(out_det['index'])
    print(f"  Output idx={out_det['index']}  shape={out_det['shape'].tolist()}  "
          f"scale={out_s:.10f}  zp={out_zp}  raw_int8={int(out_d.flatten()[0])}")

    # Gợi ý mapping
    print("\n" + "=" * 100)
    print("GỢI Ý MAPPING CHO algo_verify.py")
    print("=" * 100)
    print("Weights theo thứ tự index:")
    for i, (idx, name, shape, n_sc, _) in enumerate(weights):
        print(f"  w_list[{i}] → idx={idx}  shape={shape}  {name}")
    print("\nBiases theo thứ tự index:")
    for i, (idx, name, shape, _) in enumerate(biases):
        print(f"  b_list[{i}] → idx={idx}  shape={shape}  {name}")
    print("\nActivations theo thứ tự index:")
    for i, (idx, name, shape, scale, zp, _) in enumerate(activations):
        print(f"  act[{i}]    → idx={idx}  shape={shape}  {name}")


if __name__ == "__main__":
    main()