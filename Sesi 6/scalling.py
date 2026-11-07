import cv2
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("kp.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Scaling bebas tersehar
hasil = cv2.resize(
    img,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_LINEAR
)

# Menampilkan informasi ukuran
print("Ukuran gambar asli:", img.shape)
print("Ukuran setelah scaling:", hasil.shape)

# Menampilkan gambar
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(f"Asli: {img.shape[1]} x {img.shape[0]}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil)
plt.title(f"Scaling: {hasil.shape[1]} x {hasil.shape[0]}")
plt.axis("off")

plt.tight_layout()
plt.show()