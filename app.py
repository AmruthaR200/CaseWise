import os
import io
import base64
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename

from bson import ObjectId
from pymongo import MongoClient

from src.config import Config
from src.ml.model import load_cnn_model, preprocess_image, predict_disease
# from src.ml.explain import generate_shap_explanation, generate_lime_explanation
from src.nlp.disease_explanations import get_explanation_and_diet
from src.reports.pdf_report import build_pdf_report


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.secret_key = app.config.get("SECRET_KEY", "casewise-secret-key")

    # Ensure upload and output folders exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["XAI_FOLDER"], exist_ok=True)
    os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)

    # MongoDB client
    client = MongoClient(app.config["MONGO_URI"])
    db = client[app.config["MONGO_DB_NAME"]]
    cases_collection = db["cases"]

    # Load CNN model once at startup
    model = load_cnn_model(app.config["MODEL_PATH"], app.config["IMAGE_SIZE"], app.config["NUM_CLASSES"])

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            patient_name = request.form.get("patient_name", "").strip()
            patient_id = request.form.get("patient_id", "").strip()
            city = request.form.get("city", "").strip()
            country = request.form.get("country", "").strip()
            file = request.files.get("image")

            if not patient_name or not patient_id or not file:
                flash("Please fill all required fields and upload an image.", "danger")
                return redirect(url_for("index"))

            filename = secure_filename(file.filename)
            if filename == "":
                flash("Invalid file name.", "danger")
                return redirect(url_for("index"))

            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # Preprocess and predict
            img_array = preprocess_image(save_path, app.config["IMAGE_SIZE"])
            pred_label, pred_code, confidence, probs = predict_disease(model, img_array, app.config["DISEASE_LABELS"])

            # Generate XAI visualizations
            # shap_path = generate_shap_explanation(
            #     model, img_array, save_path, app.config["XAI_FOLDER"], prefix=patient_id + "_shap"
            # )
            # lime_path = generate_lime_explanation(
            #     model, img_array, save_path, app.config["XAI_FOLDER"], prefix=patient_id + "_lime"
            # )

            # Generate explanations and diet (English and Kannada) – disease-specific, instant lookup
            explanation_en, explanation_kn, diet_en, diet_kn = get_explanation_and_diet(pred_label)
            shap_path = None
            lime_path = None

            # Doctor suggestions (only for Mysore)
            doctor_suggestions = []
            if city.lower() == "mysore":
                doctor_suggestions = Config.MYSORE_DOCTORS.get(pred_label, Config.DEFAULT_MYSORE_DOCTORS)

            # Persist case in MongoDB
            case_doc = {
                "patient_name": patient_name,
                "patient_id": patient_id,
                "city": city,
                "country": country,
                "image_filename": filename,
                "prediction": {
                    "disease_name": pred_label,
                    "disease_code": pred_code,
                    "confidence": float(confidence),
                },
                "explanation_en": explanation_en,
                "explanation_kn": explanation_kn,
                "diet_en": diet_en,
                "diet_kn": diet_kn,
                "doctor_suggestions": doctor_suggestions,
                "shap_image": os.path.basename(shap_path) if shap_path else None,
                "lime_image": os.path.basename(lime_path) if lime_path else None,
                "created_at": datetime.utcnow(),
            }

            inserted = cases_collection.insert_one(case_doc)

            return redirect(url_for("result", case_id=str(inserted.inserted_id)))

        return render_template("index.html")

    @app.route("/result/<case_id>")
    def result(case_id):
        case = cases_collection.find_one({"_id": ObjectId(case_id)})
        if not case:
            flash("Case not found.", "danger")
            return redirect(url_for("index"))

        shap_url = None
        lime_url = None
        if case.get("shap_image"):
            shap_url = url_for("static", filename=f"xai/{case['shap_image']}")
        if case.get("lime_image"):
            lime_url = url_for("static", filename=f"xai/{case['lime_image']}")

        return render_template(
            "result.html",
            case=case,
            shap_url=shap_url,
            lime_url=lime_url,
        )

    @app.route("/report", methods=["GET", "POST"])
    def report():
        if request.method == "POST":
            patient_id = request.form.get("patient_id", "").strip()
            if not patient_id:
                flash("Please enter Patient ID.", "danger")
                return redirect(url_for("report"))

            case = cases_collection.find_one({"patient_id": patient_id}, sort=[("created_at", -1)])
            if not case:
                flash("No reports found for this Patient ID.", "warning")
                return redirect(url_for("report"))

            pdf_bytes = build_pdf_report(case)
            return send_file(
                io.BytesIO(pdf_bytes),
                as_attachment=True,
                download_name=f"CaseWise_Report_{patient_id}.pdf",
                mimetype="application/pdf",
            )

        return render_template("report.html")

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

