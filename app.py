import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==================================================
# Flask Configuration
# ==================================================

app = Flask(__name__)
app.secret_key = "super-secret-key-change-this"

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================================================
# Load Trained Model
# ==================================================

MODEL_PATH = "mobilenetv2_best.keras"
model = load_model(MODEL_PATH)

IMG_SIZE = 224

# Official 38 Classes (Alphabetical Order)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ==================================================
# Disease Information Database
# ==================================================

DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'A fungal disease that causes dark, scabby lesions on leaves and fruit, leading to premature leaf drop.',
        'organic': 'Rake and destroy infected leaves in autumn. Prune trees to improve air circulation. Apply Neem oil.',
        'chemical': 'Apply fungicides containing Captan or Myclobutanil during early spring at bud break.',
        'pesticide_name': 'Bonide Captan 50% WP Fungicide',
        'pesticide_img': 'https://m.media-amazon.com/images/I/71s8L9+wKUL._SL1500_.jpg',
        'pesticide_link': 'https://www.amazon.com/s?k=Bonide+Captan+Fungicide'
    },
    'Apple___Black_rot': {
        'description': 'A disease caused by the fungus Botryosphaeria obtusa, resulting in frog-eye leaf spots and rotting fruit.',
        'organic': 'Remove dead wood, mummies, and heavily infected branches. Ensure proper sanitation.',
        'chemical': 'Use fungicides containing Thiophanate-methyl or Captan starting at tight cluster.',
        'pesticide_name': 'Clearys 3336F Fungicide',
        'pesticide_img': 'https://m.media-amazon.com/images/I/61N9z1C65FL._SL1000_.jpg',
        'pesticide_link': 'https://www.amazon.com/s?k=Thiophanate-methyl+fungicide'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Recognized by brick-red to brown pustules appearing on both surfaces of corn leaves.',
        'organic': 'Plant rust-resistant corn varieties. Rotate crops annually to prevent fungal buildup.',
        'chemical': 'Apply foliar fungicides like Pyraclostrobin or Azoxystrobin if detected early in the season.',
        'pesticide_name': 'Headline AMP Fungicide',
        'pesticide_img': 'https://m.media-amazon.com/images/I/51A6e+m-2dL._SL1000_.jpg',
        'pesticide_link': 'https://www.amazon.com/s?k=Headline+AMP+Fungicide'
    },
    'Tomato___Early_blight': {
        'description': 'Fungal infection causing bullseye-shaped brown spots on lower leaves, eventually spreading upwards.',
        'organic': 'Mulch around the base to prevent soil splashing. Water at the base, not overhead. Use Copper soap.',
        'chemical': 'Treat with Chlorothalonil or Mancozeb-based fungicides every 7-14 days.',
        'pesticide_name': 'Daconil Fungicide (Chlorothalonil)',
        'pesticide_img': 'https://m.media-amazon.com/images/I/81kKxBwK5KL._SL1500_.jpg',
        'pesticide_link': 'https://www.amazon.com/s?k=Daconil+Fungicide'
    },
    'Tomato___Late_blight': {
        'description': 'A highly destructive disease causing dark, water-soaked spots on leaves and rapidly rotting fruit.',
        'organic': 'Destroy infected plants immediately. Do not compost. Ensure wide spacing for airflow.',
        'chemical': 'Requires aggressive treatment with Mefenoxam, Chlorothalonil, or Copper-based sprays.',
        'pesticide_name': 'Liquid Copper Fungicide',
        'pesticide_img': 'https://m.media-amazon.com/images/I/71wZ3s8O-7L._SL1500_.jpg',
        'pesticide_link': 'https://www.amazon.com/s?k=Liquid+Copper+Fungicide'
    },
    'Potato___Early_blight': {
        'description': 'Causes dark, concentric rings on older potato leaves, reducing crop yield.',
        'organic': 'Practice 3-year crop rotation. Ensure adequate soil fertility to keep plants vigorous.',
        'chemical': 'Apply Mancozeb or Chlorothalonil when symptoms first appear.',
        'pesticide_name': 'Mancozeb Flowable with Zinc',
        'pesticide_img': 'https://m.media-amazon.com/images/I/61w+T2L9xXL._SL1000_.jpg',
        'pesticide_link': 'https://www.amazon.com/s?k=Mancozeb+Fungicide'
    },
    # Default fallback for healthy plants or diseases not yet in the dictionary
    'default': {
        'description': 'Detection complete. If this is a disease, accurate identification is the first step to recovery. If marked healthy, keep up the good work!',
        'organic': 'Maintain good plant hygiene, proper watering, and adequate sunlight.',
        'chemical': 'Apply broad-spectrum preventative care if symptoms worsen. Consult local experts.',
        'pesticide_name': 'Organic Neem Oil (Broad Spectrum)',
        'pesticide_img': 'https://m.media-amazon.com/images/I/71OQ-D2n3PL._SL1500_.jpg',
        'pesticide_link': 'https://www.amazon.com/s?k=Organic+Neem+Oil+Pesticide'
    }
}

# ==================================================
# Utility Functions
# ==================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    disease_name = class_names[class_index]
    return disease_name, round(confidence, 2)

# ==================================================
# Routes
# ==================================================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected.")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

            file.save(filepath)
            disease_name, confidence = predict_disease(filepath)
            image_url = url_for("static", filename=f"uploads/{unique_filename}")

            # Fetch detailed info or use default
            info = DISEASE_INFO.get(disease_name, DISEASE_INFO['default'])

            return render_template(
                "result.html",
                disease_name=disease_name.replace("___", " - ").replace("_", " "), # Formats the name cleanly
                confidence=confidence,
                image_url=image_url,
                info=info
            )

        flash("Invalid file type. Allowed: png, jpg, jpeg.")
        return redirect(request.url)

    return render_template("upload.html")

# ==================================================
# Run Application
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)


