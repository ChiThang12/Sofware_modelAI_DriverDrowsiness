# ASIC-Optimized Driver Drowsiness Detection

Hệ thống tiền xử lý dữ liệu được tối ưu hóa đặc biệt cho thiết kế ASIC.

### 1. **Fixed-Point Arithmetic (Q0.7)**
```
Format: Q0.7 (8-bit int8)
├─ 1 bit: Sign
├─ 0 bits: Integer part
└─ 7 bits: Fractional part

Range: [0, 127] → [0, 0.9921875]
Resolution: 1/128 = 0.0078125
```

**Lợi ích:**
- ✅ Không cần FPU (Floating Point Unit)
- ✅ Chỉ dùng integer ALU
- ✅ Tiết kiệm diện tích chip
- ✅ Tiết kiệm điện năng

### 2. **Power-of-2 Dimensions**
```
Image Size: 128 x 128 (2^7 x 2^7)

Addressing: addr = (y << 7) + x
            ↑
         Chỉ cần bit-shift, không cần multiply!
```

**Lợi ích:**
- ✅ Addressing đơn giản (bit-shift)
- ✅ DMA transfer hiệu quả
- ✅ Memory tiling dễ dàng
- ✅ Pipeline-friendly

### 3. **Bit-Shift Normalization**
```c
// Normalization (uint8 -> Q0.7)
q07_value = uint8_value >> 1;  // Divide by 2

// Denormalization (Q0.7 -> uint8)
uint8_value = q07_value << 1;  // Multiply by 2

→ Không cần divider hardware!
```

### 4. **Memory Optimization**
```
Kích thước/ảnh:
- RGB float32:  252 KB
- RGB uint8:     63 KB
- Gray float32:  84 KB
- ASIC Q0.7:     16 KB  ← 16x nhỏ hơn RGB float32!

SRAM required: 64 KB (input + intermediate buffers)
```

## 📁 Cấu trúc Files

### Core Files (ASIC-specific):
- **`config.py`** - Cấu hình ASIC (Q0.7, power-of-2, etc.)
- **`preprocessing.py`** - ASIC-friendly preprocessing functions
- **`preprocess_data.py`** - Main preprocessing script
- **`split_data.py`** - Train/test splitting
- **`run_all.py`** - Master script
- **`colab_loader.py`** - Colab loader với Q0.7 conversion

### Support Files:
- **`requirements.txt`** - Dependencies
- **`INSTALL.md`** - Hướng dẫn cài đặt
- **`test_mediapipe.py`** - Test MediaPipe

## 🚀 Quick Start

### Bước 1: Cài đặt
```bash
cd D:\PROJECTDriverDrowsiness\ModelAiFinal\pre_processing
pip install -r requirements.txt
python test_mediapipe.py
```

### Bước 2: Chạy ASIC preprocessing
```bash
python run_all_asic.py
```

Output → `D:\PROJECTDriverDrowsiness\PreprocessedData_ASIC\`

### Bước 3: Upload lên Google Drive
Upload 4 files:
- `X_train_asic.npy`
- `X_test_asic.npy`
- `y_train_asic.npy`
- `y_test_asic.npy`

### Bước 4: Train trong Google Colab
```python
# Load ASIC data
from colab_loader_asic import load_asic_data
X_train, X_test, y_train, y_test = load_asic_data()

# Build model (128x128x1 input!)
model = Sequential([
    Input(shape=(128, 128, 1)),  # Power-of-2, grayscale
    Conv2D(32, (3, 3), activation='relu'),
    # ... your architecture
])

# Train
model.fit(X_train, y_train, ...)

# Convert to TFLite INT8 (ASIC-ready)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.int8
tflite_model = converter.convert()
```

## 📊 Data Format Chi Tiết

### Input Format (từ preprocessing):
```
Shape: (N, 128, 128)
Dtype: int8
Range: [0, 127]
Format: Q0.7 fixed-point
Memory: 16 KB per image
```

### Conversion to Float (cho training):
```python
# Q0.7 [0, 127] → float32 [0, 1]
X_float = X_int8.astype(np.float32) / 128.0

# Add channel dimension
X_float = np.expand_dims(X_float, axis=-1)  # (N, 128, 128, 1)
```

### Conversion back to Q0.7 (cho inference):
```python
# float32 [0, 1] → Q0.7 [0, 127]
X_int8 = np.round(X_float * 128.0).astype(np.int8)
```
