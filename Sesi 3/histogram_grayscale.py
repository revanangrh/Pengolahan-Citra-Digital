import imageio.v3 as iio
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ====== PARAMETER (silakan diubah) ======
PATH = "FotoCerah.jpeg"
BETA_TETAP = -40      # nilai beta tetap
TARGET_MEAN = 127     # target rata-rata intensitas untuk beta otomatis
GAMMA = 1.5
CLIP_LIMIT = 2.0      # CLAHE
TILE_GRID = (8, 8)    # CLAHE

# ====== BACA GAMBAR (RGB) ======
img = iio.imread(PATH)[..., :3]          # uint8, RGB


# ====== FUNGSI BANTU ======
def to_gray(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)


def adjust_beta(rgb, beta):
    """g(x) = f(x) + beta, lalu di-clip ke 0-255."""
    return np.clip(rgb.astype(np.int16) + beta, 0, 255).astype(np.uint8)


def auto_beta(rgb, target_mean=127):
    """Hitung beta otomatis supaya rata-rata gray mendekati target_mean."""
    beta = int(round(target_mean - to_gray(rgb).mean()))
    return adjust_beta(rgb, beta), beta


def gamma_correct(rgb, gamma):
    """g(x) = 255 * (f(x)/255) ^ gamma, pakai lookup table."""
    lut = (255 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return lut[rgb]


def clahe_rgb(rgb, clip=2.0, tile=(8, 8)):
    """CLAHE pada channel L (kecerahan) agar warna tidak rusak."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def histogram(gray):
    """Jumlah piksel untuk tiap intensitas 0-255."""
    return np.bincount(gray.ravel(), minlength=256)


# ====== PROSES ======
img_beta_otomatis, beta_auto = auto_beta(img, TARGET_MEAN)

hasil = [
    ("1. Asli", img),
    (f"2. Beta Tetap ({BETA_TETAP})", adjust_beta(img, BETA_TETAP)),
    (f"3. Beta Otomatis ({beta_auto:+d})", img_beta_otomatis),
    (f"4. Gamma ({GAMMA})", gamma_correct(img, GAMMA)),
    ("5. CLAHE", clahe_rgb(img, CLIP_LIMIT, TILE_GRID)),
]

# ====== PLOT PERBANDINGAN ======
fig, axes = plt.subplots(2, 5, figsize=(20, 9),
                         gridspec_kw={"height_ratios": [3, 2]})

for i, (judul, gambar) in enumerate(hasil):
    # baris atas: gambar
    axes[0, i].imshow(gambar)
    axes[0, i].set_title(judul, fontsize=13, fontweight="bold")
    axes[0, i].axis("off")

    # baris bawah: histogram grayscale
    h = histogram(to_gray(gambar))
    axes[1, i].bar(np.arange(256), h, width=1.0, color="#1f77b4", alpha=0.8)
    axes[1, i].set_title(f"Hist: {judul}", fontsize=10)
    axes[1, i].set_xlim(0, 255)
    axes[1, i].grid(True, linestyle="--", alpha=0.4)
    axes[1, i].set_xlabel("Intensitas (0-255)")
    if i == 0:
        axes[1, i].set_ylabel("Jumlah piksel")

    print(f"{judul:28s} mean = {to_gray(gambar).mean():6.2f} | "
          f"intensitas dominan = {h.argmax()} ({h.max()} piksel)")

plt.tight_layout()
plt.savefig("perbandingan.png", dpi=150)
plt.show()