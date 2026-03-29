from typing import Tuple

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


_tokenizer = None
_model = None


def _load_bert():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_name = "bert-base-multilingual-cased"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForMaskedLM.from_pretrained(model_name)
    return _tokenizer, _model


def _refine_with_bert(text: str) -> str:
    """
    Feed a sentence with a [MASK] token through BERT
    and substitute one word, giving a touch of learned phrasing.
    """
    tokenizer, model = _load_bert()
    if "[MASK]" not in text:
        return text

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    if len(mask_token_index) == 0:
        return text

    mask_index = mask_token_index[0].item()
    mask_logits = logits[0, mask_index]
    top_token_id = int(torch.argmax(mask_logits))

    new_ids = inputs.input_ids[0].clone()
    new_ids[mask_index] = top_token_id
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_explanation_and_diet(disease_name: str) -> Tuple[str, str, str, str]:
    """
    Return (explanation_en, explanation_kn, diet_en, diet_kn) with
    disease-specific content refined by BERT for English.
    """
    dn = disease_name.replace("_", " ")

    # English explanations, disease-specific (first sentence or two)
    base_explanations_en = {
        "Atelectasis": "This X-ray shows areas of [MASK] collapse in part of the lung, a condition called atelectasis. When part of the lung is not fully inflated, it can reduce oxygen levels and cause breathing difficulty.",
        "Brain_Tumor": "This scan suggests an abnormal [MASK] growth inside the brain, which may represent a brain tumour. Tumours can press on normal brain tissue and may cause headaches, seizures, or weakness.",
        "Cardiomegaly": "The heart silhouette appears [MASK] larger than normal, a finding called cardiomegaly. An enlarged heart can occur when the heart muscle has to work harder over time.",
        "Consolidation": "Parts of the lung appear more [MASK] solid and white, a sign of consolidation. This often happens when the air spaces fill with fluid, pus, or cells during infection or inflammation.",
        "Edema": "There is a pattern suggesting extra [MASK] fluid in or around the lungs, called pulmonary edema. This can be related to heart problems or other conditions that cause fluid overload.",
        "Effusion": "The image shows extra [MASK] fluid collecting around the lungs, known as a pleural effusion. This fluid can compress the lung and make breathing uncomfortable.",
        "Emphysema": "The lungs appear [MASK] over-inflated with areas of tissue damage, consistent with emphysema. This long-term damage reduces the lung's ability to move oxygen into the blood.",
        "Fibrosis": "There are streaky, [MASK] scar-like changes in the lung tissue, suggesting pulmonary fibrosis. Scarring makes the lungs stiff and can lead to chronic breathlessness.",
        "Hernia": "The image suggests that [MASK] abdominal structures may be protruding into the chest cavity, consistent with a hernia. This can change normal organ position and sometimes affect breathing.",
        "Infiltration": "The lungs show [MASK] patchy white areas called infiltrates. These can be caused by infections, inflammation, or other lung conditions.",
        "Mass": "There is a [MASK] distinct rounded shadow in the image, which may represent a mass or growth. Further tests are usually needed to know its exact nature.",
        "No_Brain_Finding": "This scan does not show a clear abnormal [MASK] lesion in the brain. However, imaging does not replace a full clinical examination by a neurologist.",
        "No_Lung_Finding": "On this X-ray, there is no obvious [MASK] abnormality seen in the lungs. Even when images look normal, symptoms should still be discussed with a doctor.",
        "Nodule": "A small [MASK] round spot is visible in the lung called a nodule. Most nodules are harmless, but some need follow-up imaging to be sure.",
        "Pleural": "The lining around the lungs, called the pleura, looks [MASK] thickened or irregular. Pleural changes can be due to past infection, inflammation, or other disease.",
        "Pneumonia": "This X-ray shows [MASK] cloudy areas in part of the lung that suggest pneumonia, an infection of the air spaces. Pneumonia can cause fever, cough, and breathing difficulty.",
        "Pneumothorax": "The image suggests [MASK] air has leaked into the space around the lung, called a pneumothorax (collapsed lung). This can cause sudden chest pain and shortness of breath.",
        "Tuberculosis": "This scan shows [MASK] changes in the lungs that are consistent with tuberculosis (TB). TB is a long-term infection that usually needs several months of medicine.",
    }

    # English diet recommendations, disease-specific (first sentence)
    base_diet_en = {
        "Atelectasis": "Take small, frequent meals that are easy to chew, drink enough warm water, and avoid very heavy or oily food that can make breathing feel harder.",
        "Brain_Tumor": "Eat soft, nutrient-rich food like vegetables, fruits, whole grains, and adequate protein; avoid skipping meals and limit very sugary or junk food.",
        "Cardiomegaly": "Prefer low-salt meals, more fruits and vegetables, lean protein, and avoid very salty, fried, or processed foods to reduce strain on the heart.",
        "Consolidation": "Take warm fluids, soups, and light home-cooked food. Avoid very cold drinks and heavy oily meals while you are recovering.",
        "Edema": "Limit added salt, avoid very salty snacks and pickles, drink water as advised by your doctor, and include fresh fruits and vegetables.",
        "Effusion": "Have light, balanced meals with less oil and salt. Avoid alcohol and smoking, and follow any fluid restriction your doctor suggests.",
        "Emphysema": "Take small, frequent meals, include proteins like pulses and eggs, avoid heavy fried foods, and never smoke or use tobacco.",
        "Fibrosis": "Eat a balanced diet with fresh fruits, vegetables, and protein. Avoid smoking, alcohol, and very processed foods.",
        "Hernia": "Prefer soft, low-spice food, avoid overeating, and avoid lying down immediately after meals. Reduce very spicy and acidic foods that cause reflux.",
        "Infiltration": "Take warm fluids, homemade soups, and simple cooked food. Avoid street food and very oily or spicy items.",
        "Mass": "Focus on nutritious, home-cooked meals with plenty of vegetables and fruits, and limit processed meats, deep-fried items, and sugary drinks.",
        "No_Brain_Finding": "Maintain a balanced diet with whole grains, fruits, vegetables, and adequate hydration to support general brain and body health.",
        "No_Lung_Finding": "Continue a healthy diet rich in fruits, vegetables, and adequate water, and avoid smoking and unnecessary alcohol.",
        "Nodule": "Choose fresh, less processed foods, lean proteins, and plenty of fruits and vegetables. Avoid smoking and limit alcohol.",
        "Pleural": "Prefer light, low-oil meals, avoid very salty and packaged foods, and drink clean water according to your doctor's advice.",
        "Pneumonia": "Take warm fluids, light easily digestible food like khichdi or soups, and avoid cold drinks or very oily items.",
        "Pneumothorax": "Eat small, gentle meals that do not cause bloating, avoid carbonated drinks, and never smoke or use tobacco.",
        "Tuberculosis": "Take protein-rich food (dal, eggs, milk), fresh fruits and vegetables, and plenty of clean water to support long-term treatment.",
    }

    # Extra generic text to make explanations multi-line
    extra_expl_en = (
        " Your doctor will confirm this finding using a full clinical examination "
        "and, if needed, further tests such as blood tests or advanced scans."
        " Getting medical advice early helps prevent complications and ensures you receive the right treatment."
    )

    extra_diet_en = (
        " Adjust this diet to your appetite, and follow any special instructions already given by your doctor."
        " If you have other conditions like diabetes, kidney or heart disease, discuss a personalised diet plan with a dietitian."
    )

    template_en = base_explanations_en.get(
        disease_name,
        f"This image shows patterns that may indicate {dn}. Please consult a specialist doctor for confirmation.",
    )
    expl_en = _refine_with_bert(template_en) + extra_expl_en

    base_diet_value = base_diet_en.get(
        disease_name,
        "Follow a balanced diet with enough fluids, fruits, and vegetables, and avoid smoking and alcohol.",
    )
    diet_en = base_diet_value + extra_diet_en

    # Kannada explanations and diets (disease-specific where possible, then extra generic text)
    explanations_kn = {
        "Atelectasis": "ಈ ಎಕ್ಸ್-ರೇ ಚಿತ್ರದಲ್ಲಿ ಶ್ವಾಸಕೋಶದ ಕೆಲವು ಭಾಗಗಳು ಸಂಪೂರ್ಣವಾಗಿ ಉಬ್ಬದೆ ಕುಗ್ಗಿರುವಂತೆ ಕಾಣಿಸುತ್ತಿವೆ. ಈ ಸ್ಥಿತಿಯನ್ನು ಅತಿಲೆಕ್ಟಾಸಿಸ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಇದರಿಂದ ಉಸಿರಾಟದಲ್ಲಿ ತೊಂದರೆ ಉಂಟಾಗಬಹುದು.",
        "Brain_Tumor": "ಈ ಸ್ಕ್ಯಾನ್‌ನಲ್ಲಿ ಮೆದುಳಿನ ಒಳಭಾಗದಲ್ಲಿ ಅಸಹಜವಾಗಿ ಬೆಳೆಯುತ್ತಿರುವ ಒಂದು ಗುಡ್ಡೆಯಂತಹ ಭಾಗ ಕಾಣುತ್ತದೆ. ಇದನ್ನು ಬ್ರೇನ್ ಟ್ಯೂಮರ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ತಲೆನೋವು, ಚಕ್ರಭ್ರಮೆ, ದುರ್ಬಲತೆ ಇತ್ಯಾದಿ ಲಕ್ಷಣಗಳನ್ನು ಉಂಟುಮಾಡಬಹುದು.",
        "Cardiomegaly": "ಚಿತ್ರದಲ್ಲಿ ಹೃದಯದ ಗಾತ್ರ ಸಾಮಾನ್ಯಕ್ಕಿಂತ ದೊಡ್ಡದಾಗಿ ಕಾಣುತ್ತದೆ. ಹೃದಯಕ್ಕೆ ಹೆಚ್ಚು ಕೆಲಸ ಬರುವ ದೀರ್ಘಕಾಲದ ಒತ್ತಡದಿಂದ ಕಾರ್ಡಿಯೋಮೆಗಲಿ (ಹೃದಯ ವೃದ್ಧಿ) ಉಂಟಾಗಬಹುದು.",
        "Consolidation": "ಶ್ವಾಸಕೋಶದ ಕೆಲವು ಭಾಗಗಳು ಬಿಳಿಯಾಗಿ ಘನವಾಗಿರುವಂತೆ ಕಾಣುತ್ತಿವೆ. ಇದನ್ನು ಕಾಂಸಾಲಿಡೇಶನ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಸಾಮಾನ್ಯವಾಗಿ ಸೋಂಕು ಅಥವಾ ಉರಿಯೂತದ ಸಮಯದಲ್ಲಿ ಗಾಳಿಯ ಬದಲು ದ್ರವ/ಪುಸ್ ತುಂಬಿದಾಗ ಉಂಟಾಗುತ್ತದೆ.",
        "Edema": "ಶ್ವಾಸಕೋಶದೊಳಗೆ ಅಥವಾ ಸುತ್ತಮುತ್ತ ಹೆಚ್ಚುವರಿ ದ್ರವ ಸಂಗ್ರಹವಾಗಿರುವಂತೆ ಚಿತ್ರದಲ್ಲಿ ಕಾಣುತ್ತದೆ. ಈ ಪರಿಸ್ಥಿತಿಯನ್ನು ಪಲ್ಮನರಿ ಎಡಿಮಾ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಹೃದಯ ಸಮಸ್ಯೆಗಳು ಅಥವಾ ದ್ರವದ ಹೆಚ್ಚುವರಿ ಕಾರಣದಿಂದ ಉಂಟಾಗಬಹುದು.",
        "Effusion": "ಶ್ವಾಸಕೋಶದ ಸುತ್ತಲಿನ ಶೂನ್ಯದಲ್ಲಿ ಹೆಚ್ಚುವರಿ ದ್ರವ ಸಂಗ್ರಹವಾಗಿದೆ. ಇದನ್ನು ಪ್ಲೀುರಲ್ ಎಫ್ಯೂಷನ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಇದರಿಂದ ಉಸಿರಾಟದ ಸಮಯದಲ್ಲಿ ತೊಂದರೆ, ನೋವು ಉಂಟಾಗಬಹುದು.",
        "Emphysema": "ಶ್ವಾಸಕೋಶಗಳು ಹೆಚ್ಚು ಉಬ್ಬಿಕೊಂಡಿರುವಂತೆ ಮತ್ತು ಒಳಗಿನ ಹಲವಾರು ಭಾಗಗಳಲ್ಲಿ ಹಾನಿ ಕಂಡುಬರುತ್ತದೆ. ಇದನ್ನು ಎಂಪೈಸೀಮಾ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ದೀರ್ಘಕಾಲದ ಧೂಮಪಾನದಿಂದ ಉಂಟಾಗುವ ಸಿಒಪಿಡಿಯ ಒಂದು ಭಾಗವಾಗಿದೆ.",
        "Fibrosis": "ಶ್ವಾಸಕೋಶದೊಳಗೆ ಚುಕ್ಕಿ-ರೇಖೆಗಳಂತಿರುವ ಕಟ್ಟಿ ಹೋಗಿದ scar ಬದಲಾವಣೆಗಳು ಕಾಣುತ್ತಿವೆ. ಇದನ್ನು ಫೈಬ್ರೋಸಿಸ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಶ್ವಾಸಕೋಶವನ್ನು ಗಟ್ಟಿ ಮಾಡುತ್ತದೆ, ಉಸಿರಾಟ ಕಷ್ಟವಾಗಬಹುದು.",
        "Hernia": "ಕೆಲವು ಹೊಟ್ಟೆ ಅಂಗಾಂಶಗಳು ಮೂಳೆಗಳ ಮಧ್ಯೆ ಮೇಲಕ್ಕೆ ಬಂದು ಎದೆಬಾಗದ ಒಳಗೆ ಕಾಣುತ್ತಿರುವಂತೆ ಚಿತ್ರದಲ್ಲಿ ಕಾಣಬಹುದು. ಇದನ್ನು ಹೆರ್ನಿಯಾ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಕೆಲವರಿಗೆ ನೋವು ಅಥವಾ ಉಸಿರಾಟದ ತೊಂದರೆ ಉಂಟುಮಾಡಬಹುದು.",
        "Infiltration": "ಶ್ವಾಸಕೋಶದೊಳಗೆ ಚುಕ್ಕಿ-ಚುಕ್ಕಿಯಾಗಿ ಬಿಳಿ ಪಾಚೆಗಳಂತೆ ಕಾಣುವ ಭಾಗಗಳನ್ನು ಇನ್‌ಫಿಲ್ಟ್ರೇಷನ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ. ಇದಕ್ಕೆ ಸೋಂಕು, ಉರಿಯೂತ ಅಥವಾ ಇತರೆ ಕಾಯಿಲೆಗಳು ಕಾರಣವಾಗಿರಬಹುದು.",
        "Mass": "ಚಿತ್ರದಲ್ಲಿ ಸ್ಪಷ್ಟವಾದ ಒಂದು ಗುಂಪು ಅಥವಾ ಗಡ್ಡೆಯಂತಹ ಬಿಳಿ ಭಾಗ ಕಂಡು ಬರುತ್ತಿದೆ. ಇದನ್ನು ಮಾಸ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಅದರ ಸ್ವಭಾವ ತಿಳಿದುಕೊಳ್ಳಲು ಹೆಚ್ಚಿನ ಪರೀಕ್ಷೆಗಳು ಅಗತ್ಯವಿರಬಹುದು.",
        "No_Brain_Finding": "ಈ ಸ್ಕ್ಯಾನ್‌ನಲ್ಲಿ ಮೆದುಳಿನೊಳಗೆ ಯಾವುದೇ ಸ್ಪಷ್ಟ ಅಸಹಜ ಗಡ್ಡೆ ಅಥವಾ ರಕ್ತಸ್ರಾವ ಕಂಡುಬಂದಿಲ್ಲ. ಆದರೂ, ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ವೈದ್ಯರಿಗೆ ವಿವರಿಸಿ ಸೂಕ್ತ ತಪಾಸಣೆ ಮಾಡಿಸಿಕೊಂಡು ಹೋಗುವುದು ಮುಖ್ಯ.",
        "No_Lung_Finding": "ಈ ಎಕ್ಸ್-ರೇ ಚಿತ್ರದಲ್ಲಿ ಶ್ವಾಸಕೋಶಗಳಲ್ಲಿ ಯಾವುದೇ ಸ್ಪಷ್ಟ ಅಸಹಜ ಬದಲಾವಣೆಗಳು ಕಾಣಿಸುತ್ತಿಲ್ಲ. ಇಮೇಜ್ ನಾರ್ಮಲ್ ಇದ್ದರೂ ಸಹ, ನಿಮ್ಮ ತೊಂದರೆಯನ್ನು ವೈದ್ಯರು ಚೆನ್ನಾಗಿ ಪರೀಕ್ಷಿಸಿ ನೋಡುವುದು ಅಗತ್ಯ.",
        "Nodule": "ಶ್ವಾಸಕೋಶದೊಳಗೆ ಒಂದು ಚಿಕ್ಕ ಗುಡ್ಡೆ/ಚುಕ್ಕಿ ಬಿಂದು ಕಂಡು ಬರುತ್ತಿದೆ, ಇದನ್ನು ನಾಡ್ಯೂಲ್ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ. ಬಹುತೇಕ ನಾಡ್ಯೂಲ್‌ಗಳು ಹಾನಿಕಾರಕವಾಗಿರುವುದಿಲ್ಲ, ಆದರೆ ಕೆಲವಕ್ಕೆ ಹಂತ ಹಂತವಾಗಿ ಫಾಲೋ-ಅಪ್ ಅಗತ್ಯವಿರಬಹುದು.",
        "Pleural": "ಶ್ವಾಸಕೋಶಗಳನ್ನು ಹೊದಿರುವ ಪ್ಲೀೂರಾ ಪದರದಲ್ಲಿ ದಪ್ಪವಾಗಿರುವ ಅಥವಾ ಅಸಮತೋಲನ ಬದಲಾವಣೆಗಳು ಕಾಣುತ್ತಿವೆ. ಇದು ಹಳೆಯ ಸೋಂಕು, ಉರಿಯೂತ ಅಥವಾ ಇತರ ಕಾರಣಗಳಿಂದ ಉಂಟಾಗಿರಬಹುದು.",
        "Pneumonia": "ಈ ಎಕ್ಸ್-ರೇ ಚಿತ್ರದಲ್ಲಿ ಶ್ವಾಸಕೋಶದ ಕೆಲ ಭಾಗಗಳು ಬಿಳಿಯಾಗಿ ದ್ರವದಿಂದ ತುಂಬಿರುವಂತೆ ಕಾಣುತ್ತಿವೆ. ಇದನ್ನು ನಿಯುಮೋನಿಯಾ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ, ಇದು ಶ್ವಾಸಕೋಶದ ಇನ್ಫೆಕ್ಷನ್ ಆಗಿದ್ದು ಜ್ವರ, ಕೆಮ್ಮು, ಉಸಿರಾಟದ ಕಾಂಚುಲು ಉಂಟುಮಾಡಬಹುದು.",
        "Pneumothorax": "ಚಿತ್ರದಲ್ಲಿ ಶ್ವಾಸಕೋಶದ ಸುತ್ತಲಿನ ಜಾಗದಲ್ಲಿ ಗಾಳಿ ಸೇರುವ ಲಕ್ಷಣಗಳು ಕಾಣುತ್ತಿವೆ. ಇದನ್ನು ನಿಯುಮೋಥೋರಾಕ್ಸ್ (ಕುಗ್ಗಿದ ಶ್ವಾಸಕೋಶ) ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ತುರ್ತು ಚಿಕಿತ್ಸೆಗೆ ಒಳಪಟ್ಟ ಸ್ಥಿತಿ ಆಗಿರಬಹುದು.",
        "Tuberculosis": "ಈ ಎಕ್ಸ್-ರೇ ಚಿತ್ರದಲ್ಲಿ ಟ್ಯೂಬರ್ಕುಲೋಸಿಸ್ (ಕ್ಷಯರೋಗ) ಗೆ ಹೊಂದುವಂತಹ ಬದಲಾವಣೆಗಳು ಶ್ವಾಸಕೋಶಗಳಲ್ಲಿ ಕಾಣಿಸುತ್ತಿವೆ. ಟಿಬಿ ಒಂದು ದೀರ್ಘಕಾಲದ ಸೋಂಕು ಆಗಿದ್ದು ಸರಿಯಾದ ಔಷಧಿಗಳನ್ನು ಹಲವು ತಿಂಗಳುಗಳ ಕಾಲ ತೆಗೆದುಕೊಳ್ಳಬೇಕು.",
    }

    diets_kn = {
        "Atelectasis": "ಸುಲಭವಾಗಿ ಜೀರ್ಣವಾಗುವ ಆಹಾರಗಳನ್ನು ಸ್ವಲ್ಪಸ್ವಲ್ಪವಾಗಿ ಸೇವಿಸಿ, ಬಿಸಿ ನೀರು ಅಥವಾ ಸೂಪ್‌ಗಳನ್ನು ಕುಡಿಯಿರಿ ಮತ್ತು ತುಂಬಾ ಎಣ್ಣೆಯುಕ್ತ, ಭಾರವಾದ ತಿಂಡಿಗಳನ್ನು ತಪ್ಪಿಸಿ.",
        "Brain_Tumor": "ಪೋಷಕಾಂಶಯುಕ್ತ ಹಣ್ಣುಗಳು, ತರಕಾರಿಗಳು, ಸಂಪೂರ್ಣ ಧಾನ್ಯಗಳು ಮತ್ತು ಪ್ರೋಟೀನ್ ಇರುವ ಆಹಾರಗಳನ್ನು ಸೇವಿಸಿ. ಜಂಕ್ ಫುಡ್ ಮತ್ತು ಹೆಚ್ಚು ಸಕ್ಕರೆ ಇರುವ ಪದಾರ್ಥಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ.",
        "Cardiomegaly": "ಉಪ್ಪು ಕಡಿಮೆ ಇರುವ ಊಟ, ಹೆಚ್ಚು ಹಣ್ಣು-ತರಕಾರಿಗಳು, ಕಡಿಮೆ ಎಣ್ಣೆಯಂಶ ಇರುವ ಪ್ರೋಟೀನ್ ಆಹಾರಗಳನ್ನು ಸೇವಿಸಿ. ಪಾಕೆಟ್ ಸ್ನ್ಯಾಕ್ಸ್, ಡೀಪ್ ಫ್ರೈಡ್ ಮತ್ತು ತೀವ್ರ ಉಪ್ಪಿನ ಪದಾರ್ಥಗಳನ್ನು ತಪ್ಪಿಸಿ.",
        "Consolidation": "ಬಿಸಿ ಸೂಪ್, ಸಾರು, ಹಿತವಾದ ಅಕ್ಕಿ ಮತ್ತು ದಳಸಿನಂತಹ ಲಘು ಆಹಾರಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ. ತಂಪು ಪಾನೀಯಗಳು ಮತ್ತು ತುಂಬಾ ಎಣ್ಣೆಯುಕ್ತ ತಿಂಡಿಗಳನ್ನು ಸಾಧ್ಯವಾದಷ್ಟು ತಪ್ಪಿಸಿ.",
        "Edema": "ಅತಿಯಾದ ಉಪ್ಪು, ಪಾಕೆಟ್ ಚಿಪ್ಸ್, ಪಾಪಡಗಳು ಇತ್ಯಾದಿಗಳನ್ನು ತಪ್ಪಿಸಿ. ವೈದ್ಯರು ಸೂಚಿಸಿದಷ್ಟು ಮಾತ್ರ ನೀರು ಕುಡಿಯಿರಿ ಮತ್ತು ಹಣ್ಣು-ತರಕಾರಿಗಳಂತಹ ತಾಜಾ ಆಹಾರಗಳನ್ನು ಹೆಚ್ಚು ಸೇವಿಸಿ.",
        "Effusion": "ಹಗುರವಾದ, ಕಡಿಮೆ ಎಣ್ಣೆಯ ಆಹಾರ, ಮನೆ ಮಾಡಿದ ಊಟ, ಹಣ್ಣು-ತರಕಾರಿಗಳನ್ನು ಸೇವಿಸಿ. ಮದ್ಯಪಾನ, ಧೂಮಪಾನ ಮತ್ತು ತುಂಬಾ ಉಪ್ಪಿನ ಪದಾರ್ಥಗಳನ್ನು ದೂರವಿಡಿ.",
        "Emphysema": "ಸಣ್ಣ ಸಣ್ಣ ಪ್ರಮಾಣದಲ್ಲಿ ಊಟ ಮಾಡಿ, ಪ್ರೋಟೀನ್ (ಕಾಳುಗಳು, ಮೊಟ್ಟೆ, ಹಾಲು) ಒಳಗೊಂಡ ಆಹಾರಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ. ಧೂಮಪಾನ ಸಂಪೂರ್ಣವಾಗಿ ನಿಲ್ಲಿಸಿ ಮತ್ತು ತುಂಬಾ ಎಣ್ಣೆಯುಕ್ತ ತಿಂಡಿಗಳನ್ನು ತಪ್ಪಿಸಿ.",
        "Fibrosis": "ಹಣ್ಣು, ತರಕಾರಿ, ಸಂಪೂರ್ಣ ಧಾನ್ಯ ಮತ್ತು ಪ್ರೋಟೀನ್‌ ಸಮೃದ್ಧ ಆಹಾರಗಳನ್ನು ಸೇವಿಸಿ. ಧೂಮಪಾನ, ಮದ್ಯಪಾನ ಮತ್ತು ತುಂಬಾ ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿದ (ಪ್ರೊಸೆಸ್ಟ್) ಆಹಾರಗಳನ್ನು ಸಾಧ್ಯವಾದಷ್ಟು ದೂರವಿಡಿ.",
        "Hernia": "ಹಗುರವಾದ, ಕಡಿಮೆ ಮಸಾಲೆಯ ಆಹಾರ, ಜಾಸ್ತಿ ಹೊಟ್ಟೆ ತುಂಬಿಸಿಕೊಳ್ಳದಂತೆ ಸಣ್ಣ ಸಣ್ಣ ಊಟ ಮಾಡಿ. ಊಟದ ತಕ್ಷಣ ಮಲಗುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಜಾಸ್ತಿ ಕಾರಂ, ಆಮ್ಲೀಯ ಆಹಾರಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ.",
        "Infiltration": "ಬಿಸಿ ನೀರು, ಸೂಪ್‌, ಇಡಿಯಪ್ಪಂ, ಇಡ್ಲಿ ಮುಂತಾದ ಸಾದಾ ತಿಂಡಿಗಳನ್ನು ಸೇವಿಸಿ. ರಸ್ತೆಯಲ್ಲಿನ ಆಹಾರ, ತುಂಬಾ ಎಣ್ಣೆ ಮತ್ತು ಮಸಾಲೆಯ ಆಹಾರಗಳನ್ನು ತಪ್ಪಿಸುವುದು ಒಳಿತು.",
        "Mass": "ತಾಜಾ ಹಣ್ಣು-ತರಕಾರಿಗಳು, ಧಾನ್ಯ ಮತ್ತು ಪ್ರೋಟೀನ್‌ ಸಮೃದ್ಧ ಮನೆದಿನ ಊಟ ಮುಖ್ಯ. ಡೀಪ್ ಫ್ರೈಡ್, ಹೆಚ್ಚು ಉಪ್ಪು-ಸಕ್ಕರೆ ಇರುವ ಪದಾರ್ಥಗಳು ಹಾಗೂ ಜಂಕ್ ಫುಡ್ ಅನ್ನು ತಪ್ಪಿಸಿ.",
        "No_Brain_Finding": "ಸಾಧಾರಣವಾಗಿ ಆರೋಗ್ಯಕರ ಆಹಾರ ಕ್ರಮವನ್ನು ಅನುಸರಿಸಿ: ಹಣ್ಣು, ತರಕಾರಿ, ಸಂಪೂರ್ಣ ಧಾನ್ಯಗಳು ಮತ್ತು ಸಾಕಷ್ಟು ನೀರು. ಅತಿಯಾದ ಜಂಕ್ ಫುಡ್ ಮತ್ತು ಸಕ್ಕರೆ ತುಂಬಿದ ಪಾನೀಯಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ.",
        "No_Lung_Finding": "ಹಣ್ಣು-ತರಕಾರಿ, ಸಮತೋಲನಯುತ ಊಟ ಮತ್ತು ಸಾಕಷ್ಟು ನೀರು ಸೇವಿಸಿ. ಧೂಮಪಾನ, ಮದ್ಯಪಾನ ಮತ್ತು ಅನಾವಶ್ಯಕ ರಾಸಾಯನಿಕ ವಾಸನೆಗಳನ್ನು ದೂರವಿಡಿ.",
        "Nodule": "ತಾಜಾ, ಪ್ರಕ್ರಿಯೆಗೊಳಿಸದ ಆಹಾರ, ಪ್ರೋಟೀನ್ ಮತ್ತು ಹಣ್ಣು-ತರಕಾರಿಗಳನ್ನು ಹೆಚ್ಚು ಸೇವಿಸಿ. ಧೂಮಪಾನ ಮತ್ತು ಮದ್ಯಪಾನವನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ತಪ್ಪಿಸಿ.",
        "Pleural": "ಕಡಿಮೆ ಉಪ್ಪು ಮತ್ತು ಎಣ್ಣೆಯಿರುವ ಮನೆದಿನ ಊಟ, ಹಣ್ಣು-ತರಕಾರಿಗಳನ್ನು ಸೇವಿಸಿ. ಹೆಚ್ಚು ಪ್ಯಾಕೆಜ್ಡ್ ಸ್ನ್ಯಾಕ್ಸ್ ಮತ್ತು ಅತಿಯಾದ ಉಪ್ಪಿನ ಪದಾರ್ಥಗಳನ್ನು ದೂರವಿಡಿ.",
        "Pneumonia": "ಬಿಸಿ ದ್ರವಪದಾರ್ಥಗಳು, ಸಾರು, ಸೂಪ್‌, ಖಿಚಡಿ ಮುಂತಾದ ಸುಲಭವಾಗಿ ಜೀರ್ಣವಾಗುವ ಆಹಾರಗಳನ್ನು ಸೇವಿಸಿ. ತಂಪು ಪಾನೀಯಗಳು ಮತ್ತು ತುಂಬಾ ಎಣ್ಣೆಯುಕ್ತ ತಿಂಡಿಗಳನ್ನು ತಪ್ಪಿಸಿ.",
        "Pneumothorax": "ಹೊಟ್ಟೆ ತುಂಬಾ ಗಾಳಿ ಏರುವುದು ತಪ್ಪಿಸಲು ಗ್ಯಾಸ್ ಉಂಟುಮಾಡುವ ಮತ್ತು ಕಾರ್ಬೊನೇಟೆಡ್ ಪಾನೀಯಗಳನ್ನು ತಪ್ಪಿಸಿ. ಹಗುರವಾದ ಮನೆದಿನ ಊಟವನ್ನು ಸಣ್ಣ ಪ್ರಮಾಣದಲ್ಲಿ ಸೇವಿಸಿ.",
        "Tuberculosis": "ಪ್ರೋಟೀನ್‌ ಸಮೃದ್ಧ ಆಹಾರ (ಡಾಲ್, ಮೊಟ್ಟೆ, ಹಾಲು ಉತ್ಪನ್ನಗಳು), ಹಣ್ಣು-ತರಕಾರಿಗಳನ್ನು ಹೆಚ್ಚಾಗಿ ಸೇವಿಸಿ ಮತ್ತು ಸಾಕಷ್ಟು ಸ್ವಚ್ಛ ನೀರು ಕುಡಿಯಿರಿ. ಊಟ ತಪ್ಪಿಸದೇ ನಿಯಮಿತ ಸಮಯಕ್ಕೆ ಮಾಡಿಕೊಳ್ಳಿ.",
    }

    extra_expl_kn = (
        " ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಮತ್ತು ಆರೋಗ್ಯ ಇತಿಹಾಸವನ್ನು ವೈದ್ಯರಿಗೆ ವಿವರವಾಗಿ ತಿಳಿಸಿ ಮತ್ತು ಅವರ ಸಲಹೆಯಂತೆ ಅಗತ್ಯವಾದ ಮುಂದಿನ ಪರೀಕ್ಷೆಗಳನ್ನು ಮಾಡಿಸಿಕೊಳ್ಳುವುದು ಅತ್ಯಂತ ಮುಖ್ಯ."
        " ತ್ವರಿತವಾಗಿ ಚಿಕಿತ್ಸೆ ಆರಂಭಿಸಿದರೆ ಗಂಭೀರ ತೊಂದರೆಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಬಹುದು."
    )

    extra_diet_kn = (
        " ನಿಮ್ಮ ಶರೀರದ ಸ್ಥಿತಿ, ತೂಕ ಮತ್ತು ಇತರ ಕಾಯಿಲೆಗಳ ಆಧಾರದ ಮೇಲೆ ವೈದ್ಯರು ಅಥವಾ ಡಯಟಿಷಿಯನ್ ಸೂಚಿಸಿದ ವಿಶೇಷ ಆಹಾರ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ."
        " ಊಟ ತಪ್ಪಿಸದೇ, ಸಣ್ಣ ಪ್ರಮಾಣದ, ನಿಯಮಿತ ಸಮಯದ ಆಹಾರ ಸೇವನೆ ದೇಹವನ್ನು ಮರುಸ್ಥಾಪಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ."
    )

    explanation_kn_base = explanations_kn.get(
        disease_name,
        f"ಈ ಚಿತ್ರದಲ್ಲಿ {dn} ಗೆ ಹೊಂದುವ ಕೆಲವು ಲಕ್ಷಣಗಳು ಕಾಣುತ್ತಿವೆ. ದಯವಿಟ್ಟು ತಜ್ಞ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಅವರ ಸಲಹೆ ಪ್ರಕಾರ ಮುಂದಿನ ತಪಾಸಣೆಗಳನ್ನು ಮಾಡಿಸಿಕೊಳ್ಳಿ.",
    )
    explanation_kn = explanation_kn_base + extra_expl_kn

    diet_kn_base = diets_kn.get(
        disease_name,
        "ಸಮತೋಲನಯುತ ಆಹಾರ, ಹಣ್ಣುಗಳು, ತರಕಾರಿಗಳು ಮತ್ತು ಸಾಕಷ್ಟು ನೀರು ಸೇವಿಸಿ. ಧೂಮಪಾನ ಮತ್ತು ಮದ್ಯಪಾನವನ್ನು ತಪ್ಪಿಸಿ.",
    )
    diet_kn = diet_kn_base + extra_diet_kn

    return expl_en, explanation_kn, diet_en, diet_kn


