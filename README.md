# Sofware_modelAI_DriverDrowsiness
ASIC-Optimized Driver Drowsiness Detection
Hệ thống tiền xử lý dữ liệu được tối ưu hóa đặc biệt cho thiết kế ASIC.
🎯 Tối ưu hóa cho ASIC
1. Fixed-Point Arithmetic (Q0.7)
Format: Q0.7 (8-bit int8)
├─ 1 bit: Sign
├─ 0 bits: Integer part
└─ 7 bits: Fractional part

Range: [0, 127] → [0, 0.9921875]
Resolution: 1/128 = 0.0078125
Lợi ích:

✅ Không cần FPU (Floating Point Unit)
✅ Chỉ dùng integer ALU
✅ Tiết kiệm diện tích chip
✅ Tiết kiệm điện năng

2. Power-of-2 Dimensions
Image Size: 128 x 128 (2^7 x 2^7)

Addressing: addr = (y << 7) + x
            ↑
         Chỉ cần bit-shift, không cần multiply!
Lợi ích:

✅ Addressing đơn giản (bit-shift)
✅ DMA transfer hiệu quả
✅ Memory tiling dễ dàng
✅ Pipeline-friendly

3. Bit-Shift Normalization
c// Normalization (uint8 -> Q0.7)
q07_value = uint8_value >> 1;  // Divide by 2

// Denormalization (Q0.7 -> uint8)
uint8_value = q07_value << 1;  // Multiply by 2

→ Không cần divider hardware!
4. Memory Optimization
Kích thước/ảnh:
- RGB float32:  252 KB
- RGB uint8:     63 KB
- Gray float32:  84 KB
- ASIC Q0.7:     16 KB  ← 16x nhỏ hơn RGB float32!

SRAM required: 64 KB (input + intermediate buffers)
📁 Cấu trúc Files
Core Files (ASIC-specific):

config_asic.py - Cấu hình ASIC (Q0.7, power-of-2, etc.)
preprocessing_asic.py - ASIC-friendly preprocessing functions
preprocess_data_asic.py - Main preprocessing script
split_data_asic.py - Train/test splitting
run_all_asic.py - Master script
colab_loader_asic.py - Colab loader với Q0.7 conversion
ASIC_DESIGN_GUIDE.md - Hướng dẫn thiết kế ASIC chi tiết
