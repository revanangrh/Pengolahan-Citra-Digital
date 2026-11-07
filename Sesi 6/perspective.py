
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. MEMBACA GAMBAR
# ==================================================

nama_file = "dokumen.jpeg"

img = cv2.imread(nama_file)

if img is None:
    raise FileNotFoundError(
        f"Gambar '{nama_file}' tidak ditemukan. "
        "Periksa nama file dan lokasi gambar."
    )

# Mengubah BGR menjadi RGB
img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

# Mengambil ukuran gambar asli
tinggi, lebar = img.shape[:2]

print("Ukuran gambar asli:")
print("Lebar  :", lebar)
print("Tinggi :", tinggi)


# ==================================================
# 2. MENYIAPKAN GAMBAR UNTUK TAMPILAN
# ==================================================

titik = []

# Batas maksimal ukuran jendela
max_lebar = 800
max_tinggi = 600

# Menghitung faktor pengecilan
skala = min(
    max_lebar / lebar,
    max_tinggi / tinggi,
    1.0
)

# Memperkecil gambar hanya untuk tampilan
display = cv2.resize(
    img,
    None,
    fx=skala,
    fy=skala,
    interpolation=cv2.INTER_AREA
)

img_display = display.copy()


# ==================================================
# 3. FUNGSI MEMILIH 4 TITIK DENGAN MOUSE
# ==================================================

def pilih_titik(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(titik) < 4:

            # Mengubah koordinat tampilan
            # menjadi koordinat gambar asli
            x_asli = int(round(x / skala))
            y_asli = int(round(y / skala))

            # Memastikan koordinat tidak keluar gambar
            x_asli = min(max(x_asli, 0), lebar - 1)
            y_asli = min(max(y_asli, 0), tinggi - 1)

            titik.append([x_asli, y_asli])

            print(
                f"Titik {len(titik)}: "
                f"({x_asli}, {y_asli})"
            )

            # Menandai titik pada gambar tampilan
            cv2.circle(
                img_display,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )

            # Memberikan nomor pada titik
            cv2.putText(
                img_display,
                str(len(titik)),
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "Pilih 4 Sudut Dokumen",
                img_display
            )


# ==================================================
# 4. MENAMPILKAN GAMBAR DAN MEMILIH TITIK
# ==================================================

nama_window = "Pilih 4 Sudut Dokumen"

cv2.namedWindow(
    nama_window,
    cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback(
    nama_window,
    pilih_titik
)

cv2.imshow(
    nama_window,
    img_display
)

print("\nPilih empat sudut dokumen secara berurutan:")
print("1. Kiri atas")
print("2. Kanan atas")
print("3. Kanan bawah")
print("4. Kiri bawah")
print("\nTekan ESC untuk membatalkan.")

# Menunggu hingga empat titik dipilih
while len(titik) < 4:

    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # Tombol ESC
        break

cv2.destroyAllWindows()

# Validasi jumlah titik
if len(titik) != 4:
    raise RuntimeError(
        "Pemilihan dibatalkan. "
        "Anda harus memilih empat titik."
    )


# ==================================================
# 5. MENENTUKAN TITIK SUMBER
# ==================================================

pts1 = np.float32(titik)

print("\nKoordinat titik sumber:")
print(pts1)


# ==================================================
# 6. MENGHITUNG UKURAN HASIL TRANSFORMASI
# ==================================================

# Menghitung panjang sisi atas
lebar_atas = np.linalg.norm(
    pts1[1] - pts1[0]
)

# Menghitung panjang sisi bawah
lebar_bawah = np.linalg.norm(
    pts1[2] - pts1[3]
)

# Mengambil lebar terbesar
lebar_baru = int(
    max(lebar_atas, lebar_bawah)
)

# Menghitung panjang sisi kiri
tinggi_kiri = np.linalg.norm(
    pts1[3] - pts1[0]
)

# Menghitung panjang sisi kanan
tinggi_kanan = np.linalg.norm(
    pts1[2] - pts1[1]
)

# Mengambil tinggi terbesar
tinggi_baru = int(
    max(tinggi_kiri, tinggi_kanan)
)

# Validasi ukuran
if lebar_baru < 1 or tinggi_baru < 1:
    raise ValueError(
        "Ukuran hasil transformasi tidak valid."
    )

print("\nUkuran hasil transformasi:")
print("Lebar  :", lebar_baru)
print("Tinggi :", tinggi_baru)


# ==================================================
# 7. MENENTUKAN TITIK TUJUAN
# ==================================================

pts2 = np.float32([
    [0, 0],
    [lebar_baru - 1, 0],
    [lebar_baru - 1, tinggi_baru - 1],
    [0, tinggi_baru - 1]
])


# ==================================================
# 8. MENGHITUNG MATRIKS PERSPEKTIF
# ==================================================

M = cv2.getPerspectiveTransform(
    pts1,
    pts2
)

print("\nMatriks Perspektif:")
print(M)


# ==================================================
# 9. MELAKUKAN PERSPECTIVE TRANSFORMATION
# ==================================================

hasil = cv2.warpPerspective(
    img_rgb,
    M,
    (lebar_baru, tinggi_baru)
)


# ==================================================
# 10. MENAMPILKAN HASIL
# ==================================================

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil)
plt.title("Hasil Perspective Transformation")
plt.axis("off")

plt.tight_layout()
plt.show()


# ==================================================
# 11. MENYIMPAN HASIL
# ==================================================

# Mengubah RGB menjadi BGR sebelum disimpan
hasil_bgr = cv2.cvtColor(
    hasil,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    "hasil_perspective.jpg",
    hasil_bgr
)

print("\nTransformasi berhasil!")
print("Hasil disimpan sebagai hasil_perspective.jpg")
