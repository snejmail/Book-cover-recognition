import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers, optimizers
from utils import process_image


dataset_path = r"C:\SNEJI\Docs\Work related\ML and AI\Book Covers Recognition\archive (1)\book-covers"
categories = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

print(f"Found {len(categories)} categories.")

image_paths = []
labels = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    if os.path.isdir(folder):
        images = os.listdir(folder)
        print(f"{category}: {len(images)} images")

        for i, image_name in enumerate(images[:3]):
            img_path = os.path.join(folder, image_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_paths.append(img_path)
            labels.append(category)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels_encoded, test_size=0.2, random_state=42
)

print(f"Training samples: {len(train_paths)}")
print(f"Test samples: {len(test_paths)}")

train_images = []
for path in train_paths:
    processed = process_image(path)
    if processed is not None:
        train_images.append(processed)
train_images = np.array(train_images)

test_images = []
for path in test_paths:
    processed = process_image(path)
    if processed is not None:
        test_images.append(processed)
test_images = np.array(test_images)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])

model.compile(optimizer=optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, test_labels)
)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy Over Epochs')
plt.show()
