"""
Peningkatan Kualitas Citra menggunakan imageio, numpy, dan scipy.ndimage
Mata Kuliah: Pengolahan Citra Digital
"""
import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")          # hapus baris ini jika ingin plt.show() di komputer sendiri
import matplotlib.pyplot as plt

PATH = "FotoCerah.jpeg"

# ---------- 1. Baca gambar & ubah ke grayscale ----------
img = iio.imread(PATH)[..., :3]
gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])  # float


def to_u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def histogram(a):
    return np.bincount(to_u8(a).ravel(), minlength=256)


# ---------- 2. Peningkatan kontras (numpy) ----------
def contrast_stretch(a, p_low=1, p_high=99):
    """Regangkan kontras: persentil p_low..p_high dipetakan ke 0..255."""
    lo, hi = np.percentile(a, [p_low, p_high])
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255)


def hist_equalization(a):
    """Histogram equalization manual memakai CDF."""
    a8 = to_u8(a)
    h = np.bincount(a8.ravel(), minlength=256)
    cdf = h.cumsum()
    cdf_min = cdf[cdf > 0].min()
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255)
    return lut[a8]


def gamma(a, g=1.5):
    return 255 * (a / 255.0) ** g


# ---------- 3. Filter spasial (scipy.ndimage) ----------
def denoise_median(a, size=3):
    return ndi.median_filter(a, size=size)


def smooth_gaussian(a, sigma=1.5):
    return ndi.gaussian_filter(a, sigma=sigma)


def sharpen_unsharp(a, sigma=2.0, amount=1.5):
    """Unsharp masking: citra + amount * (citra - citra_blur)."""
    blur = ndi.gaussian_filter(a, sigma=sigma)
    return a + amount * (a - blur)


def edge_sobel(a):
    gx = ndi.sobel(a, axis=1)
    gy = ndi.sobel(a, axis=0)
    mag = np.hypot(gx, gy)
    return mag / mag.max() * 255


# ---------- 4. Terapkan ----------
hasil = [
    ("Asli (grayscale)",            gray),
    ("Gamma 1.5",                   gamma(gray, 1.5)),
    ("Contrast Stretching",         contrast_stretch(gray)),
    ("Histogram Equalization",      hist_equalization(gray)),
    ("Median Filter 3x3",           denoise_median(gray, 3)),
    ("Gaussian Blur (sigma 1.5)",   smooth_gaussian(gray, 1.5)),
    ("Unsharp Masking",             sharpen_unsharp(contrast_stretch(gray))),
    ("Deteksi Tepi Sobel",          edge_sobel(smooth_gaussian(gray, 1.0))),
]

# ---------- 5. Statistik ----------
print(f"{'Metode':28s} {'mean':>7s} {'std (kontras)':>14s} {'dominan':>8s}")
for nama, a in hasil:
    a8 = to_u8(a)
    print(f"{nama:28s} {a8.mean():7.2f} {a8.std():14.2f} {histogram(a).argmax():8d}")

# ---------- 6. Plot perbandingan (grid 4 kolom, 2 blok) ----------
kolom = 4
blok = (len(hasil) + kolom - 1) // kolom
fig, axes = plt.subplots(blok * 2, kolom, figsize=(4.5 * kolom, 9 * blok / 1.2),
                         gridspec_kw={"height_ratios": [3, 2] * blok})
for k, (nama, a) in enumerate(hasil):
    r, c = (k // kolom) * 2, k % kolom
    axes[r, c].imshow(to_u8(a), cmap="gray", vmin=0, vmax=255)
    axes[r, c].set_title(nama, fontsize=11, fontweight="bold")
    axes[r, c].axis("off")
    axes[r + 1, c].bar(np.arange(256), histogram(a), width=1.0, color="#1f77b4")
    axes[r + 1, c].set_xlim(0, 255)
    axes[r + 1, c].set_title("Histogram", fontsize=9)
    axes[r + 1, c].grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("hasil_program.png", dpi=110)
# plt.show()