import numpy as np
import pickle
from tensorflow.keras.models import load_model
from utils import process_image

best_model = load_model('model/best_model.h5')

with open('model/categories.pkl', 'rb') as f:
    categories = pickle.load(f)


def predict_book_cover(img_path):
    img = process_image(img_path)
    if img is None:
        return "Invalid image", 0.0

    img = np.expand_dims(img, axis=0)

    predictions = best_model.predict(img)
    predicted_index = np.argmax(predictions)
    genre_label = categories[predicted_index]
    confidence = predictions[0][predicted_index]

    return genre_label, confidence

