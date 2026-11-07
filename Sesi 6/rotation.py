import cv2
import matplotlib.pyplot as plt

img = cv2.imread("kp.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tinggi, lebar = img.shape[:2]

# Menentukan titik pusat
pusat = (lebar // 2, tinggi // 2)

# Rotasi 45 derajat
M = cv2.getRotationMatrix2D(
    pusat, 45, 1.0
)

hasil = cv2.warpAffine(
    img, M, (lebar, tinggi)
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hasil)
plt.title("Rotasi 45 Derajat")
plt.axis("off")

plt.show()