import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar seperti biasa
img = cv2.imread("kp.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tinggi, lebar = img.shape[:2]

# ini proses translasi
M = np.float32([
    [1, 0, 150],
    [0, 1, 75]
])

# Melakukan translasi
hasil = cv2.warpAffine(
    img, M, (lebar, tinggi)
)

# Menampilkan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil)
plt.title("Hasil Translation")
plt.axis("off")

plt.show()