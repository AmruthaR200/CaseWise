import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Image size must match Config.IMAGE_SIZE
IMG_WIDTH, IMG_HEIGHT = 150, 150

BATCH_SIZE = 48
EPOCHS = 30

CLASS_NAMES = [
    "Atelectasis", "Brain_Tumor", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No_Brain_Finding", "No_Lung_Finding", "Nodule", "Pleural", "Pneumonia",
    "Pneumothorax", "Tuberculosis",
]

# Model definition
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(len(CLASS_NAMES), activation="softmax"))  # multi-class

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Data generators
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
)

validation_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
)

# Train
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
)

# Save model where the web app expects it
model_path = os.path.join(MODEL_DIR, "casewise_cnn.h5")
model.save(model_path)
print(f"Model saved to {model_path}")