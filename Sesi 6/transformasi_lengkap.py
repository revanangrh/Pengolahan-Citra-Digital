import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. MEMBACA CITRA ASLI
# ==================================================
nama_file = "mas gibran.webp"
img = cv2.imread(nama_file)

if img is None:
    raise FileNotFoundError(
        f"Gambar '{nama_file}' tidak ditemukan. "
        "Pastikan gambar berada dalam folder yang sama."
    )

# Mengubah format warna BGR ke RGB untuk visualisasi Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tinggi, lebar = img_rgb.shape[:2]
pusat = (lebar // 2, tinggi // 2)

# ==================================================
# 2. IMPLEMENTASI TRANSLATION
# ==================================================
# Mensejajarkan/memindahkan citra sejauh tx=50 piksel ke kanan dan ty=30 piksel ke bawah
M_trans = np.float32([
    [1, 0, 50],
    [0, 1, 30]
])
img_translation = cv2.warpAffine(img_rgb, M_trans, (lebar, tinggi))

# ==================================================
# 3. IMPLEMENTASI SCALING (0.5x dan 2.0x)
# ==================================================
img_scaling_05 = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
img_scaling_20 = cv2.resize(img_rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

# ==================================================
# 4. IMPLEMENTASI ROTATION (45°, 90°, dan 180°)
# ==================================================
M_rot_45 = cv2.getRotationMatrix2D(pusat, 45, 1.0)
img_rot_45 = cv2.warpAffine(img_rgb, M_rot_45, (lebar, tinggi))

M_rot_90 = cv2.getRotationMatrix2D(pusat, 90, 1.0)
img_rot_90 = cv2.warpAffine(img_rgb, M_rot_90, (lebar, tinggi))

M_rot_180 = cv2.getRotationMatrix2D(pusat, 180, 1.0)
img_rot_180 = cv2.warpAffine(img_rgb, M_rot_180, (lebar, tinggi))

# ==================================================
# 5. IMPLEMENTASI FLIPPING (Horizontal & Vertical)
# ==================================================
img_flip_h = cv2.flip(img_rgb, 1)  # 1 = Horizontal Flip
img_flip_v = cv2.flip(img_rgb, 0)  # 0 = Vertical Flip

# ==================================================
# 6. MENAMPILKAN CITRA ASLI DAN SELURUH HASIL TRANSFORMASI
# ==================================================
daftar_gambar = [
    (img_rgb, "Citra Asli"),
    (img_translation, "Translation (tx=50, ty=30)"),
    (img_scaling_05, f"Scaling 0.5x\n({img_scaling_05.shape[1]}x{img_scaling_05.shape[0]})"),
    (img_scaling_20, f"Scaling 2.0x\n({img_scaling_20.shape[1]}x{img_scaling_20.shape[0]})"),
    (img_rot_45, "Rotation 45°"),
    (img_rot_90, "Rotation 90°"),
    (img_rot_180, "Rotation 180°"),
    (img_flip_h, "Horizontal Flip"),
    (img_flip_v, "Vertical Flip")
]

plt.figure(figsize=(15, 12))

for i, (gambar, judul) in enumerate(daftar_gambar, 1):
    plt.subplot(3, 3, i)
    plt.imshow(gambar)
    plt.title(judul, fontsize=11, fontweight='bold')
    plt.axis("off")

plt.tight_layout()
plt.show()