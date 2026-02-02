# ASIC Design Guide for Driver Drowsiness Detection

## 🎯 Overview

Hướng dẫn chi tiết thiết kế ASIC cho hệ thống phát hiện buồn ngủ khi lái xe.

## 📊 Data Format Specification

### Fixed-Point Format: Q0.7

```
┌─────────────────────────────────────┐
│  Sign | Integer | Fractional        │
│   1   |    0    |      7            │
│  bit  |  bits   |    bits           │
└─────────────────────────────────────┘
    ↓       ↓           ↓
   S.IIIIIII.FFFFFFF (total: 8 bits)
   0.0000000.0000000 = 0
   0.0000000.1111111 = 127/128 ≈ 0.992
```

**Properties:**
- **Range**: [0, 127] (unsigned) or [-128, 127] (signed)
- **Resolution**: 1/128 = 0.0078125
- **Max value**: 127/128 ≈ 0.9921875
- **Perfect for normalized images [0, 1]**

### Conversion Formulas

```c
// uint8 [0, 255] → Q0.7 [0, 127]
q07_value = uint8_value >> 1;

// Q0.7 [0, 127] → float [0, 1]
float_value = (float)q07_value / 128.0f;

// Q0.7 [0, 127] → uint8 [0, 255]
uint8_value = q07_value << 1;
```

## 🔧 ASIC Architecture Recommendations

### 1. Memory Organization

```
┌─────────────────────────────────────────────────┐
│           INPUT BUFFER (16 KB)                  │
│         128 x 128 pixels x 1 byte               │
│                                                 │
│  Addressing: addr = (y << 7) + x                │
│  (left shift by 7 = multiply by 128)            │
└─────────────────────────────────────────────────┘

Memory Map:
0x0000 - 0x3FFF: Input image buffer (16 KB)
0x4000 - 0x7FFF: Intermediate results
0x8000 - 0x8FFF: Weights (quantized)
0x9000 - 0x9FFF: Output buffer
```

**Benefits:**
- Power-of-2 addressing (simple logic)
- No multiply needed for indexing
- Easy DMA transfer (aligned)

### 2. Datapath Design

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Input   │   │   Conv   │   │  Pool    │   │   FC     │
│  Buffer  │──▶│  Engine  │──▶│  Engine  │──▶│  Engine  │
│  (int8)  │   │  (int8)  │   │  (int8)  │   │  (int8)  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
     ▲              │              │              │
     │              ▼              ▼              ▼
     │         ┌────────────────────────────────┐
     └─────────│    Control Unit (FSM)          │
               └────────────────────────────────┘
```

**Key Components:**

#### a) MAC (Multiply-Accumulate) Unit
```verilog
module mac_int8 (
    input  signed [7:0] a,     // Input activation
    input  signed [7:0] w,     // Weight
    input  signed [31:0] acc,  // Accumulator
    output signed [31:0] result
);
    assign result = acc + (a * w);
endmodule
```

#### b) Quantization Unit
```verilog
module quantize_relu (
    input  signed [31:0] acc,     // 32-bit accumulator
    input  [7:0] scale,           // Scale factor
    output signed [7:0] out       // 8-bit output
);
    wire signed [31:0] scaled = acc >> scale;
    assign out = (scaled > 127) ? 127 :
                 (scaled < 0) ? 0 : scaled[7:0];
endmodule
```

### 3. Convolution Engine

```
┌───────────────────────────────────────────────────┐
│          Convolution Engine (3x3 kernel)          │
│                                                   │
│  ┌─────┐ ┌─────┐ ┌─────┐                         │
│  │ MAC │ │ MAC │ │ MAC │  ──┐                     │
│  └─────┘ └─────┘ └─────┘    │                     │
│  ┌─────┐ ┌─────┐ ┌─────┐    ├──▶  Accumulator   │
│  │ MAC │ │ MAC │ │ MAC │  ──┤                     │
│  └─────┘ └─────┘ └─────┘    │                     │
│  ┌─────┐ ┌─────┐ ┌─────┐    │                     │
│  │ MAC │ │ MAC │ │ MAC │  ──┘                     │
│  └─────┘ └─────┘ └─────┘                          │
│                                                   │
│  9 parallel MAC units (int8 x int8)               │
└───────────────────────────────────────────────────┘
```

**Throughput:**
- 9 MACs/cycle
- 128x128 input, 3x3 kernel
- Cycles needed ≈ (126x126) / 9 ≈ 1,764 cycles/channel

### 4. Memory Bandwidth Requirements

```
Input:     128 x 128 x 1 byte = 16 KB
Weights:   Depends on model (e.g., 32 filters, 3x3 = 288 bytes/layer)
Output:    126 x 126 x 32 = 508 KB (first layer example)

Bandwidth = (Input + Weights + Output) / Time
          = (16 KB + 0.3 KB + 508 KB) / (1764 cycles / 200MHz)
          ≈ 60 GB/s (peak)

Practical: ~10 GB/s with pipelining and reuse
```

## 💡 Hardware Optimizations

### 1. Weight Quantization

```python
# Quantize weights to int8
def quantize_weights(weights_float):
    """
    Quantize float32 weights to int8
    
    Args:
        weights_float: Float weights [-1, 1]
    
    Returns:
        weights_int8: int8 weights [-128, 127]
        scale: Scale factor for dequantization
    """
    # Find max absolute value
    max_val = np.max(np.abs(weights_float))
    
    # Scale factor
    scale = 127.0 / max_val
    
    # Quantize
    weights_int8 = np.round(weights_float * scale).astype(np.int8)
    
    return weights_int8, scale
```

### 2. Activation Quantization

```c
// ReLU with int8
int8_t relu_int8(int32_t x) {
    if (x < 0) return 0;
    if (x > 127) return 127;
    return (int8_t)x;
}

// Batch normalization (fused with quantization)
int8_t batch_norm_relu(int32_t acc, int8_t mean, int8_t inv_std, int8_t beta) {
    // BN: y = (x - mean) * inv_std + beta
    int32_t normalized = ((acc - mean) * inv_std) >> 7;  // Divide by 128
    int32_t shifted = normalized + beta;
    return relu_int8(shifted);
}
```

### 3. Pipelining Strategy

```
Pipeline Stage:
┌────────┬────────┬────────┬────────┬────────┐
│ Fetch  │  MAC   │  Acc   │  ReLU  │ Write  │
└────────┴────────┴────────┴────────┴────────┘
    1        1        1        1        1    cycles

5-stage pipeline: 1 result/cycle (after fill)
```

## 🎨 RTL Design Example

### Top-level Module

```verilog
module drowsiness_detector_asic (
    // Clock and reset
    input  wire         clk,
    input  wire         rst_n,
    
    // Input interface
    input  wire [7:0]   pixel_in,
    input  wire         pixel_valid,
    output wire         pixel_ready,
    
    // Output interface
    output wire [7:0]   result,
    output wire         result_valid,
    input  wire         result_ready,
    
    // Control
    input  wire         start,
    output wire         done
);

// Internal signals
reg [7:0] image_buffer [0:16383];  // 128x128 buffer
reg [13:0] pixel_count;
reg [2:0] state;

// State machine
localparam IDLE       = 3'd0;
localparam LOAD_IMAGE = 3'd1;
localparam CONV_LAYER = 3'd2;
localparam POOL_LAYER = 3'd3;
localparam FC_LAYER   = 3'd4;
localparam OUTPUT     = 3'd5;

// State machine implementation
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        pixel_count <= 0;
    end else begin
        case (state)
            IDLE: begin
                if (start) state <= LOAD_IMAGE;
            end
            
            LOAD_IMAGE: begin
                if (pixel_valid && pixel_ready) begin
                    image_buffer[pixel_count] <= pixel_in;
                    pixel_count <= pixel_count + 1;
                    if (pixel_count == 16383) begin
                        state <= CONV_LAYER;
                        pixel_count <= 0;
                    end
                end
            end
            
            CONV_LAYER: begin
                // Convolution engine operation
                if (conv_done) state <= POOL_LAYER;
            end
            
            POOL_LAYER: begin
                // Pooling operation
                if (pool_done) state <= FC_LAYER;
            end
            
            FC_LAYER: begin
                // Fully connected layer
                if (fc_done) state <= OUTPUT;
            end
            
            OUTPUT: begin
                if (result_ready) state <= IDLE;
            end
        endcase
    end
end

endmodule
```

## 📈 Performance Estimation

### Power Budget

```
Component          Power (mW)    Activity
─────────────────────────────────────────
MAC Units (64)     40           80%
Memory (SRAM)      25           60%
Control Logic      10           100%
I/O                15           50%
Clock Tree         10           100%
─────────────────────────────────────────
TOTAL              100 mW
```

### Throughput Calculation

```
Image size: 128 x 128 = 16,384 pixels
Layers: 5 (example CNN)

Cycles per layer:
- Conv1: 16,384 cycles
- Pool1: 4,096 cycles
- Conv2: 4,096 cycles
- Pool2: 1,024 cycles
- FC:    256 cycles

Total: ~26,000 cycles

@ 200 MHz: 26,000 / 200M = 130 µs per frame
Throughput: ~7,700 FPS
Latency: 130 µs
```

### Area Estimation (28nm process)

```
Component              Area (mm²)
───────────────────────────────────
MAC Units (64)         0.50
SRAM (64 KB)           1.20
Control Logic          0.30
I/O Pads               0.50
───────────────────────────────────
TOTAL (core)           2.50 mm²
TOTAL (with pad)       3.50 mm²
```

## 🔬 Testing & Verification

### 1. Fixed-Point Accuracy Test

```python
def test_fixed_point_accuracy():
    """Test accuracy loss from quantization"""
    
    # Original float values
    float_vals = np.linspace(0, 1, 256)
    
    # Convert to Q0.7
    q07_vals = np.round(float_vals * 128).astype(np.int8)
    
    # Convert back to float
    recovered = q07_vals.astype(np.float32) / 128.0
    
    # Calculate error
    error = np.abs(float_vals - recovered)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"Max error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")
    print(f"SNR: {-20*np.log10(mean_error):.2f} dB")
```

### 2. Testbench Example

```systemverilog
module tb_drowsiness_detector;

reg clk, rst_n, start;
reg [7:0] pixel_in;
reg pixel_valid;
wire pixel_ready, result_valid, done;
wire [7:0] result;

// DUT
drowsiness_detector_asic dut (.*);

// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100 MHz
end

// Test sequence
initial begin
    // Reset
    rst_n = 0;
    #100 rst_n = 1;
    
    // Load test image
    start = 1;
    #10 start = 0;
    
    // Feed pixels
    for (int i = 0; i < 16384; i++) begin
        pixel_in = test_image[i];
        pixel_valid = 1;
        @(posedge clk);
    end
    pixel_valid = 0;
    
    // Wait for result
    wait(done);
    
    $display("Result: %d", result);
    $finish;
end

endmodule
```

## 🚀 Next Steps

1. **Model Training**: Train với quantization-aware training
2. **RTL Implementation**: Implement trong Verilog/SystemVerilog
3. **Simulation**: Verify functionality với testbench
4. **Synthesis**: Synthesize với target technology library
5. **Place & Route**: Layout design
6. **Tape-out**: Final GDSII generation

## 📚 References

- Fixed-Point Arithmetic
- Quantization-Aware Training
- ASIC Design Flow
- Low-Power Design Techniques