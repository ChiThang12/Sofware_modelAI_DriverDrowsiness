"""
Viewer cho ASIC preprocessed data (.npy files)
Hiển thị ảnh mẫu từ X_train, X_test — format 64x64 int8 Q0.7
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# ===================================================================
# CẤU HÌNH — Chỉnh lại nếu cần
# ===================================================================
OUTPUT_DIR = r"D:\PROJECTDriverDrowsiness\ModelAiFinal\PreprocessedData_ASIC"

X_TRAIN_FILE = os.path.join(OUTPUT_DIR, "X_train_asic.npy")
X_TEST_FILE  = os.path.join(OUTPUT_DIR, "X_test_asic.npy")
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "y_train_asic.npy")
Y_TEST_FILE  = os.path.join(OUTPUT_DIR, "y_test_asic.npy")

CATEGORIES   = {0: "Fatigue", 1: "Active"}
SCALE_FACTOR = 128.0   # Q0.7 → float
EXPECTED_SIZE = 64     # Kích thước mong đợi sau khi update


# ===================================================================
# LOAD
# ===================================================================
def load_data():
    print("=" * 60)
    print("  LOADING .NPY FILES")
    print("=" * 60)

    files = {
        "X_train": X_TRAIN_FILE,
        "X_test" : X_TEST_FILE,
        "y_train": Y_TRAIN_FILE,
        "y_test" : Y_TEST_FILE,
    }

    data   = {}
    all_ok = True

    for name, path in files.items():
        if os.path.exists(path):
            arr = np.load(path)
            data[name] = arr
            mb = arr.nbytes / (1024 * 1024)
            print(f"  {name:8s}  shape={str(arr.shape):22s}  "
                  f"dtype={str(arr.dtype):6s}  {mb:.2f} MB")
        else:
            print(f"  NOT FOUND: {path}")
            all_ok = False

    if not all_ok:
        print("\n  Chạy run_all_asic.py trước.")
        sys.exit(1)

    # Cảnh báo nếu data vẫn là 128x128 (cache cũ)
    h = data["X_train"].shape[1]
    if h != EXPECTED_SIZE:
        print(f"\n  WARNING: Data size = {h}x{h}, expected {EXPECTED_SIZE}x{EXPECTED_SIZE}.")
        print(f"  Hãy xóa cache và chạy lại run_all_asic.py để tạo data {EXPECTED_SIZE}x{EXPECTED_SIZE}.")

    return data


# ===================================================================
# STATISTICS
# ===================================================================
def print_statistics(data):
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    h, w = X_train.shape[1], X_train.shape[2]

    print("\n" + "=" * 60)
    print("  THỐNG KÊ DATASET")
    print("=" * 60)

    print(f"\n  Shape & Format:")
    print(f"    X_train  : {X_train.shape}  →  {X_train.shape[0]} ảnh, {h}x{w} px")
    print(f"    X_test   : {X_test.shape}")
    print(f"    dtype    : {X_train.dtype}  (int8 = Q0.7 fixed-point)")
    print(f"    range    : [{X_train.min()}, {X_train.max()}]  (int8 [0..127])")
    print(f"    float    : [{X_train.min()/SCALE_FACTOR:.4f}, "
          f"{X_train.max()/SCALE_FACTOR:.4f}]  (sau /128)")
    print(f"    memory/frame: {h*w} bytes = {h*w/1024:.2f} KB")

    for split, ys in [("y_train", y_train), ("y_test", y_test)]:
        print(f"\n  Phân bố {split}:")
        for cls, cnt in zip(*np.unique(ys, return_counts=True)):
            print(f"    {CATEGORIES[cls]:12s}: {cnt:5d}  ({100*cnt/len(ys):.1f}%)")

    total_mb = (X_train.nbytes + X_test.nbytes) / (1024 * 1024)
    print(f"\n  Bộ nhớ tổng: {total_mb:.2f} MB")


# ===================================================================
# SAMPLE IMAGES
# ===================================================================
def show_sample_images(data, n_samples=5):
    """Hiển thị ảnh mẫu: n_samples Fatigue + n_samples Active."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    h, w    = X_train.shape[1], X_train.shape[2]

    n_rows = 2
    n_cols = n_samples

    fig = plt.figure(figsize=(n_cols * 2.8, n_rows * 3.2 + 1.5))
    fig.patch.set_facecolor("#1a1a2e")

    fig.suptitle(
        f"INPUT MODEL AI — ASIC Preprocessed Images\n"
        f"Format: Q0.7 fixed-point (int8)  |  Size: {h}x{w} Grayscale",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.15)

    for row, cls in enumerate([0, 1]):
        cls_idx = np.where(y_train == cls)[0]
        chosen  = np.random.choice(cls_idx,
                                   size=min(n_samples, len(cls_idx)),
                                   replace=False)
        for col, idx in enumerate(chosen):
            ax       = fig.add_subplot(gs[row, col])
            img_int8 = X_train[idx]
            img_f    = img_int8.astype(np.float32) / SCALE_FACTOR

            ax.imshow(img_f, cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"{CATEGORIES[cls]}\n"
                f"idx={idx}  [{img_int8.min()},{img_int8.max()}]",
                color="white", fontsize=7, pad=3,
            )
            ax.axis("off")
            border_color = "#e94560" if cls == 0 else "#0f9b8e"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(1.5)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="#e94560", label="Fatigue (class 0)"),
            Patch(facecolor="#0f9b8e", label="Active  (class 1)"),
        ],
        loc="lower center", ncol=2, framealpha=0.2,
        labelcolor="white", fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.savefig("sample_images.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("  Saved → sample_images.png")


# ===================================================================
# PIXEL DETAIL
# ===================================================================
def show_pixel_detail(data):
    """Heatmap giá trị int8 + histogram phân bố pixel."""
    X_train = data["X_train"]
    y_train = data["y_train"]

    fatigue_img = X_train[np.where(y_train == 0)[0][0]]
    active_img  = X_train[np.where(y_train == 1)[0][0]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor("#16213e")
    fig.suptitle("CHI TIẾT PIXEL — Input Model AI",
                 color="white", fontsize=13, fontweight="bold")

    for row, (img, label) in enumerate([(fatigue_img, "Fatigue"),
                                         (active_img,  "Active")]):
        img_f = img.astype(np.float32) / SCALE_FACTOR

        # Col 0: Ảnh grayscale
        axes[row, 0].imshow(img_f, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{label} — {img.shape[0]}x{img.shape[1]} Grayscale",
                                color="white", fontsize=10)
        axes[row, 0].axis("off")

        # Col 1: Heatmap int8 (8x8 patch góc trái trên — phù hợp 64x64)
        patch_size = 8
        patch = img[:patch_size, :patch_size].astype(np.float32)
        im = axes[row, 1].imshow(patch, cmap="RdYlGn", aspect="auto")
        axes[row, 1].set_title(
            f"{label} — Giá trị int8\n({patch_size}x{patch_size} góc trái trên)",
            color="white", fontsize=9,
        )
        axes[row, 1].tick_params(colors="gray", labelsize=7)
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)

        # Col 2: Histogram
        color = "#e94560" if row == 0 else "#0f9b8e"
        axes[row, 2].hist(img.flatten(), bins=64, range=(0, 127),
                          color=color, alpha=0.85, edgecolor="none")
        axes[row, 2].set_title(f"{label} — Histogram pixel",
                                color="white", fontsize=9)
        axes[row, 2].set_xlabel("Giá trị int8 [0..127]", color="gray", fontsize=8)
        axes[row, 2].set_ylabel("Số pixel",              color="gray", fontsize=8)
        axes[row, 2].tick_params(colors="gray")
        axes[row, 2].set_facecolor("#0d0d1a")
        axes[row, 2].spines[:].set_color("#333")

        print(f"  [{label}] mean={img_f.mean():.4f}  std={img_f.std():.4f}  "
              f"min={img.min()}  max={img.max()}")

    plt.tight_layout()
    plt.savefig("pixel_detail.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("  Saved → pixel_detail.png")


# ===================================================================
# CONVERSION DEMO
# ===================================================================
def show_conversion_demo(data):
    """Minh họa int8 → float32 → uint8 cho 1 ảnh mẫu."""
    X_train = data["X_train"]
    img_int8  = X_train[0]
    img_float = img_int8.astype(np.float32) / SCALE_FACTOR
    img_uint8 = (img_float * 255).astype(np.uint8)

    h, w = img_int8.shape

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"QUÁ TRÌNH CONVERT TRƯỚC KHI VÀO MODEL  ({h}x{w})",
        color="white", fontsize=12, fontweight="bold",
    )

    configs = [
        (img_int8,  "gray", 0, 127, f"1. Lưu trong .npy\nint8  [0 → 127]\n(Q0.7, {h}x{w})"),
        (img_float, "gray", 0, 1.0, f"2. Đưa vào Model\nfloat32  [0.0 → 1.0]\n(÷ 128)"),
        (img_uint8, "gray", 0, 255, f"3. Hiển thị mắt người\nuint8  [0 → 255]\n(× 255)"),
    ]

    for ax, (img, cmap, vmin, vmax, title) in zip(axes, configs):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color="white", fontsize=9.5, pad=6)
        ax.axis("off")

    fig.text(0.355, 0.5, "÷ 128", ha="center", va="center",
             color="#f5a623", fontsize=14, fontweight="bold")
    fig.text(0.645, 0.5, "× 255", ha="center", va="center",
             color="#7ed321", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("conversion_demo.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("  Saved → conversion_demo.png")


# ===================================================================
# RESIZE COMPARISON  (so sánh 128 vs 64 nếu có cache cũ)
# ===================================================================
def show_resize_comparison(data):
    """So sánh trực quan ảnh 64x64 vs 128x128 từ cùng 1 sample."""
    X_train = data["X_train"]
    img64   = X_train[0].astype(np.float32) / SCALE_FACTOR
    import cv2
    img128  = cv2.resize(
        (img64 * 255).astype(np.uint8),
        (128, 128), interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("So sánh kích thước — Cùng 1 ảnh",
                 color="white", fontsize=12, fontweight="bold")

    axes[0].imshow(img128, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("128×128 (cũ)\n16 KB/frame  |  vùng mắt ~16px",
                      color="white", fontsize=9, pad=6)
    axes[0].axis("off")

    axes[1].imshow(img64, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("64×64 (hiện tại)\n4 KB/frame  |  vùng mắt ~8px",
                      color="#0f9b8e", fontsize=9, pad=6)
    axes[1].axis("off")
    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_color("#0f9b8e")
        spine.set_linewidth(2)

    # Zoom vùng mắt
    eye_y1, eye_y2 = int(64 * 0.25), int(64 * 0.55)
    eye_x1, eye_x2 = int(64 * 0.10), int(64 * 0.90)
    eye_patch = img64[eye_y1:eye_y2, eye_x1:eye_x2]
    axes[2].imshow(eye_patch, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("Zoom vùng mắt (64×64)\n~8-12px mỗi mắt — đủ phân biệt mở/nhắm",
                      color="#f5a623", fontsize=9, pad=6)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("resize_comparison.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("  Saved → resize_comparison.png")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("  NPY DATA VIEWER — ASIC Driver Drowsiness  (64x64)")
    print("=" * 60)

    data = load_data()
    print_statistics(data)

    print("\n  Sample images...")
    show_sample_images(data, n_samples=5)

    print("\n  Pixel detail...")
    show_pixel_detail(data)

    print("\n  Conversion demo...")
    show_conversion_demo(data)

    print("\n  Resize comparison...")
    show_resize_comparison(data)

    print("\n" + "=" * 60)
    print("  TỔNG KẾT — INPUT VÀO MODEL AI")
    print("=" * 60)
    h = data["X_train"].shape[1]
    print(f"""
  Lưu trong .npy : int8    [0, 127]   — Q0.7 fixed-point
  Vào model      : float32 [0.0, 1.0] — X.astype(float32) / 128.0
  Shape          : (N, {h}, {h})
  Memory/frame   : {h*h} bytes = {h*h/1024:.2f} KB
  Nhãn           : 0 = Fatigue  |  1 = Active
    """)