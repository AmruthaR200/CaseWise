import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# --------- paths ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "classification_results_one.csv")  # change if needed
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 150, 150   # must match the shape stored in CSV
NUM_CLASSES = 18

CLASS_NAMES = [
    "Atelectasis", "Brain_Tumor", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No_Brain_Finding", "No_Lung_Finding", "Nodule", "Pleural", "Pneumonia",
    "Pneumothorax", "Tuberculosis",
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}

# --------- 1. load CSV ---------
df = pd.read_csv(CSV_PATH)

# Keep only rows whose True Label is in our class list
df = df[df["True Label"].isin(CLASS_TO_INDEX.keys())].reset_index(drop=True)

# --------- 2. parse Image column into arrays ---------
def parse_image(txt):
    # txt is a string like '[[[0.123 0.456 ...'
    arr = np.array(ast.literal_eval(txt), dtype="float32")
    # Ensure shape is (150,150,3); adjust/reshape if needed
    return arr

images = np.stack(df["Image"].apply(parse_image).values)  # shape: (N, H, W, C)

# --------- 3. encode labels ---------
labels_idx = df["True Label"].map(CLASS_TO_INDEX).astype(int).values
labels_one_hot = to_categorical(labels_idx, num_classes=NUM_CLASSES)

# --------- 4. train/validation split ---------
X_train, X_val, y_train, y_val = train_test_split(
    images, labels_one_hot, test_size=0.2, random_state=42, stratify=labels_idx
)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:", X_val.shape, y_val.shape)

# --------- 5. define model ---------
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --------- 6. train ---------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
)

# --------- 7. save model ---------
model_path = os.path.join(MODEL_DIR, "casewise_cnn.h5")
model.save(model_path)
print(f"Model saved to: {model_path}")