import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca Citra Asli
image_path = 'Sindoro.png'
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Mean Filter (Averaging)
mean_3x3 = cv2.blur(img_rgb, (3, 3))
mean_5x5 = cv2.blur(img_rgb, (5, 5))

# 3. Gaussian Filter
gaussian_3x3 = cv2.GaussianBlur(img_rgb, (3, 3), 0)
gaussian_5x5 = cv2.GaussianBlur(img_rgb, (5, 5), 0)

# 4. Median Filter
median_3x3 = cv2.medianBlur(img_rgb, 3)
median_5x5 = cv2.medianBlur(img_rgb, 5)

# 5. Sharpening Filter
# Menggunakan kernel Laplacian/Sharpening standar
kernel_sharpening = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
sharpened_img = cv2.filter2D(img_rgb, -1, kernel_sharpening)

# 6. Menampilkan Hasil Perbandingan dalam Grid
titles = [
    'Citra Asli', 
    'Mean Filter (3x3)', 'Mean Filter (5x5)', 
    'Gaussian Filter (3x3)', 'Gaussian Filter (5x5)', 
    'Median Filter (3x3)', 'Median Filter (5x5)', 
    'Sharpening'
]

images = [
    img_rgb, 
    mean_3x3, mean_5x5, 
    gaussian_3x3, gaussian_5x5, 
    median_3x3, median_5x5, 
    sharpened_img
]

plt.figure(figsize=(16, 10))
for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=11, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.show()