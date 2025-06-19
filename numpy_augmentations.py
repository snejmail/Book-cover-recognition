import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def show_images(original, augmented, title, filename):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(augmented)
    plt.title(title)
    plt.axis('off')

    plt.show()
    print(f"Saved {filename}")


image_path = "archive (1)/book-covers/Health/0000006.jpg"
original_img = Image.open(image_path).resize((128, 128))
original_array = np.array(original_img)

flipped = np.fliplr(original_array)
show_images(original_array, flipped, "Flipped", "flipped_comparison.png")

crop_size = 100
start_y = np.random.randint(0, original_array.shape[0] - crop_size)
start_x = np.random.randint(0, original_array.shape[1] - crop_size)
cropped = original_array[start_y:start_y+crop_size, start_x:start_x+crop_size, :]
show_images(original_array, cropped, "Cropped", "cropped_comparison.png")

brightness_factor = 1.5
brightened = np.clip(original_array * brightness_factor, 0, 255).astype(np.uint8)
show_images(original_array, brightened, "Brightened", "brightened_comparison.png")

