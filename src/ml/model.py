import os
import numpy as np
from typing import List, Tuple

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def load_cnn_model(model_path: str, image_size: Tuple[int, int], num_classes: int):
    """
    Load a pre-trained CNN model. If the model file is missing,
    build a very small placeholder model so that the app can still run for demo purposes.
    """
    if os.path.exists(model_path):
        return load_model(model_path)

    # Lightweight fallback model for demo (not for production use)
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(image_size[0], image_size[1], 3)),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def preprocess_image(img_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_disease(model, img_array: np.ndarray, labels: List[Tuple[str, str]]):
    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    disease_name, disease_code = labels[idx]
    return disease_name, disease_code, confidence, preds

