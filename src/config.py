import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    SECRET_KEY = os.getenv("SECRET_KEY", "casewise-secret-key")

    # Folders
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    XAI_FOLDER = os.path.join(BASE_DIR, "static", "xai")
    REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "casewise_db")

    # CNN model
    IMAGE_SIZE = (150, 150)
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        os.path.join(BASE_DIR, "models", "casewise_cnn.h5")
    )

    DISEASE_LABELS = [
        ("Atelectasis", "X-001"),
        ("Brain_Tumor", "X-002"),
        ("Cardiomegaly", "X-003"),
        ("Consolidation", "X-004"),
        ("Edema", "X-005"),
        ("Effusion", "X-006"),
        ("Emphysema", "X-007"),
        ("Fibrosis", "X-008"),
        ("Hernia", "X-009"),
        ("Infiltration", "X-010"),
        ("Mass", "X-011"),
        ("No_Brain_Finding", "X-012"),
        ("No_Lung_Finding", "X-013"),
        ("Nodule", "X-014"),
        ("Pleural", "X-015"),
        ("Pneumonia", "X-016"),
        ("Pneumothorax", "X-017"),
        ("Tuberculosis", "X-018"),
    ]

    NUM_CLASSES = len(DISEASE_LABELS)

    # Doctor directory for Mysuru
    MYSORE_DOCTORS = {

        "Atelectasis": [
            {
                "name": "Dr. Madhu K",
                "specialization": "Pulmonologist",
                "hospital": "Apollo BGS Hospitals, Mysuru",
                "contact": "+91-821-2568888",
            }
        ],

        "Brain_Tumor": [
            {
                "name": "Dr. Prakash Bhat",
                "specialization": "Neurosurgeon",
                "hospital": "Manipal Hospitals, Mysuru",
                "contact": "+91-821-4006000",
            }
        ],

        "Cardiomegaly": [
            {
                "name": "Dr. Anitha Narayan",
                "specialization": "Cardiologist",
                "hospital": "Narayana Multispeciality Hospital, Mysuru",
                "contact": "+91-821-7122222",
            }
        ],

        "Consolidation": [
            {
                "name": "Dr. Lakshmi Narasimhan R",
                "specialization": "Pulmonologist",
                "hospital": "Manipal Hospitals, Mysuru",
                "contact": "+91-821-4006000",
            }
        ],

        "Edema": [
            {
                "name": "Dr. Avinash R",
                "specialization": "Pulmonologist",
                "hospital": "JSS Hospital, Mysuru",
                "contact": "+91-821-2548416",
            }
        ],

        "Effusion": [
            {
                "name": "Dr. Madhu K",
                "specialization": "Pulmonologist",
                "hospital": "Apollo BGS Hospitals, Mysuru",
                "contact": "+91-821-2568888",
            }
        ],

        "Emphysema": [
            {
                "name": "Dr. Kavya Shetty",
                "specialization": "Pulmonologist",
                "hospital": "Apollo BGS Hospitals, Mysuru",
                "contact": "+91-821-2568888",
            }
        ],

        "Fibrosis": [
            {
                "name": "Dr. Avinash R",
                "specialization": "Pulmonologist",
                "hospital": "JSS Hospital, Mysuru",
                "contact": "+91-821-2548416",
            }
        ],

        "Hernia": [
            {
                "name": "Dr. Praveen Kumar",
                "specialization": "General Surgeon",
                "hospital": "Apollo BGS Hospitals, Mysuru",
                "contact": "+91-821-2568888",
            }
        ],

        "Infiltration": [
            {
                "name": "Dr. Madhu K",
                "specialization": "Pulmonologist",
                "hospital": "Apollo BGS Hospitals, Mysuru",
                "contact": "+91-821-2568888",
            }
        ],

        "Mass": [
            {
                "name": "Dr. Ramesh S",
                "specialization": "Oncologist",
                "hospital": "Bharath Cancer Hospital, Mysuru",
                "contact": "+91-821-2512634",
            }
        ],

        "No_Brain_Finding": [
            {
                "name": "Dr. Nandini R",
                "specialization": "Neurologist",
                "hospital": "Manipal Hospitals, Mysuru",
                "contact": "+91-821-4006000",
            }
        ],

        "No_Lung_Finding": [
            {
                "name": "Dr. Madhu K",
                "specialization": "Pulmonologist",
                "hospital": "Apollo BGS Hospitals, Mysuru",
                "contact": "+91-821-2568888",
            }
        ],

        "Nodule": [
            {
                "name": "Dr. Ramesh S",
                "specialization": "Oncologist",
                "hospital": "Bharath Cancer Hospital, Mysuru",
                "contact": "+91-821-2512634",
            }
        ],

        "Pleural": [
            {
                "name": "Dr. Lakshmi Narasimhan R",
                "specialization": "Pulmonologist",
                "hospital": "Manipal Hospitals, Mysuru",
                "contact": "+91-821-4006000",
            }
        ],

        "Pneumonia": [
            {
                "name": "Dr. Meghana Rao",
                "specialization": "Pulmonologist",
                "hospital": "Apollo BGS Hospitals, Mysuru",
                "contact": "+91-821-2568888",
            }
        ],

        "Pneumothorax": [
            {
                "name": "Dr. Avinash R",
                "specialization": "Pulmonologist",
                "hospital": "JSS Hospital, Mysuru",
                "contact": "+91-821-2548416",
            }
        ],

        "Tuberculosis": [
            {
                "name": "Dr. Sandeep Kulkarni",
                "specialization": "Chest Physician",
                "hospital": "JSS Hospital, Mysuru",
                "contact": "+91-821-2548416",
            }
        ],
    }

    # Default: return all doctors if disease not matched
    DEFAULT_MYSORE_DOCTORS = [
        doc
        for disease in MYSORE_DOCTORS.values()
        for doc in disease
    ]