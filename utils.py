import cv2


def process_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    return img

