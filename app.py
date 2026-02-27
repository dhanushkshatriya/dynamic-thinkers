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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================================================
# Load Trained Model
# ==================================================

MODEL_PATH = "mobilenetv2_best.keras"
model = load_model(MODEL_PATH)

IMG_SIZE = 224

# ==================================================
# Disease Treatment Database
# ==================================================

disease_info = {

    "Tomato___Early_blight": {
        "organic": "Apply neem oil spray weekly and remove infected leaves.",
        "chemical": "Spray Mancozeb or Chlorothalonil fungicide.",
        "pesticide_name": "Dhanuka M-45 Mancozeb",
        "pesticide_link": "https://www.amazon.in/s?k=mancozeb+fungicide"
    },

    "Tomato___Late_blight": {
        "organic": "Use compost tea spray and baking soda solution.",
        "chemical": "Apply Metalaxyl based fungicide.",
        "pesticide_name": "Ridomil Gold",
        "pesticide_link": "https://www.amazon.in/s?k=ridomil+fungicide"
    },

    "Potato___Early_blight": {
        "organic": "Apply neem oil and keep foliage dry.",
        "chemical": "Use Chlorothalonil spray.",
        "pesticide_name": "Kavach Fungicide",
        "pesticide_link": "https://www.amazon.in/s?k=kavach+fungicide"
    },

    "Potato___Late_blight": {
        "organic": "Use garlic extract spray.",
        "chemical": "Apply Metalaxyl fungicide.",
        "pesticide_name": "Ridomil Gold",
        "pesticide_link": "https://www.amazon.in/s?k=ridomil+fungicide"
    },

    "Apple___Apple_scab": {
        "organic": "Apply neem oil and prune affected leaves.",
        "chemical": "Use Captan fungicide spray.",
        "pesticide_name": "Captan Fungicide",
        "pesticide_link": "https://www.amazon.in/s?k=captan+fungicide"
    },

    "Tomato___healthy": {
        "organic": "No treatment needed. Maintain proper watering.",
        "chemical": "No chemical needed.",
        "pesticide_name": "Growth Booster",
        "pesticide_link": "https://www.amazon.in/s?k=plant+growth+booster"
    }
}

# ==================================================
# Official 38 Classes
# ==================================================

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

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

            info = disease_info.get(disease_name, {
                "organic": "Apply neem oil spray.",
                "chemical": "Use broad spectrum fungicide.",
                "pesticide_name": "Generic Fungicide",
                "pesticide_link": "https://www.amazon.in/s?k=plant+fungicide"
            })

            return render_template(
                "result.html",
                disease_name=disease_name,
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
    app.run(host="0.0.0.0", port=5000, debug=True)
