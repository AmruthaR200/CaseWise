import os
from typing import Optional

import numpy as np
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_shap_explanation(model, img_array: np.ndarray, original_path: str, out_dir: str, prefix: str) -> Optional[str]:
    """
    Generate a SHAP explanation image highlighting important regions.
    For performance, this uses a small background sample of the same image.
    """
    try:
        _ensure_dir(out_dir)
        background = img_array.copy()

        # DeepExplainer expects samples, for small demo we use the same image as background
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(img_array)

        # Plot SHAP for the predicted class (index 0 from list)
        plt.figure(figsize=(4, 4))
        shap.image_plot(shap_values, img_array, show=False)

        out_path = os.path.join(out_dir, f"{prefix}_shap.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception:
        # Fail gracefully – explanation is optional
        return None


def generate_lime_explanation(model, img_array: np.ndarray, original_path: str, out_dir: str, prefix: str) -> Optional[str]:
    """
    Generate a LIME explanation image that highlights superpixels influencing the prediction.
    """
    try:
        _ensure_dir(out_dir)

        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            images = np.array(images).astype("float32") / 255.0
            return model.predict(images)

        from tensorflow.keras.preprocessing import image as kimage

        pil_img = kimage.load_img(original_path)
        img_np = kimage.img_to_array(pil_img).astype("float32") / 255.0

        explanation = explainer.explain_instance(
            img_np.astype("double"),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=100,
        )

        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )

        plt.figure(figsize=(4, 4))
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.axis("off")

        out_path = os.path.join(out_dir, f"{prefix}_lime.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception:
        return None

