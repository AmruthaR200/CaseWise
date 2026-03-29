import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "casewise_cnn.h5")

IMG_WIDTH, IMG_HEIGHT = 150, 150

CLASS_NAMES = [
    "Atelectasis", "Brain_Tumor", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No_Brain_Finding", "No_Lung_Finding", "Nodule", "Pleural", "Pneumonia",
    "Pneumothorax", "Tuberculosis",
]

model = load_model(MODEL_PATH)

predictions = []
true_labels = []
image_names = []
images = []

for class_name in CLASS_NAMES:
    test_folder = os.path.join(DATASET_PATH, "test", class_name)
    if not os.path.isdir(test_folder):
        continue
    for image_file in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype("float32") / 255.0
        img_input = np.expand_dims(img, axis=0)

        pred = model.predict(img_input)
        predicted_class_index = int(np.argmax(pred))

        predictions.append(CLASS_NAMES[predicted_class_index])
        true_labels.append(class_name)
        image_names.append(image_file)
        images.append(img)

# Metrics
report = classification_report(true_labels, predictions, target_names=CLASS_NAMES, zero_division=0)
confusion_mat = confusion_matrix(true_labels, predictions, labels=CLASS_NAMES)

print("Classification Report:")
print(report)
print("\nConfusion Matrix:")
print(confusion_mat)

# Save CSV (without raw image arrays to keep it simple)
results_df = pd.DataFrame({
    "Image Name": image_names,
    "True Label": true_labels,
    "Predicted Label": predictions,
})
results_df.to_csv(os.path.join(BASE_DIR, "classification_results_one.csv"), index=False)
print("Results saved to classification_results_one.csv")