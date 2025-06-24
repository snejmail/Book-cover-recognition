import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage, stats

image_path = "archive (1)/book-covers/Art-Photography/0000005.jpg"
img = Image.open(image_path).resize((128, 128)).convert('L')
img_array = np.array(img)

# # Example 1: Gaussian Blur (Smoothing)

blurred = ndimage.gaussian_filter(img_array, sigma=2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred, cmap='gray')
plt.title("Gaussian Blur")
plt.axis('off')

plt.show()


# Example 2: Edge Detection (Sobel Filter)

sobel_x = ndimage.sobel(img_array, axis=0)
sobel_y = ndimage.sobel(img_array, axis=1)

edges = np.hypot(sobel_x, sobel_y)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Sobel Edges")
plt.axis('off')

plt.show()

# Example 3: Paired t-test on accuracy scores

model1_acc = np.array([0.75, 0.78, 0.74, 0.77, 0.76])
model2_acc = np.array([0.80, 0.82, 0.79, 0.81, 0.83])

t_stat, p_value = stats.ttest_rel(model1_acc, model2_acc)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference between models")
else:
    print("No significant difference between models")
