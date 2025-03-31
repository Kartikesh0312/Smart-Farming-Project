from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import json
import pickle
import pandas as pd
import numpy as np
import torchvision.models as models
import os

app = Flask(__name__)

# Load the plant disease model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet18(weights=None)
num_classes = 39
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load the irrigation model
with open("irrigation_model.pkl", "rb") as f:
    irrigation_model = pickle.load(f)

# Load the crop yield model
with open("yield_model.pkl", "rb") as f:
    yield_model = pickle.load(f)

# Function to preprocess image for disease prediction
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to predict disease
def predict_disease(image):
    image = preprocess_image(image)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

# Function to predict irrigation
def predict_irrigation(temperature, humidity, rainfall, soil_moisture):
    input_data = pd.DataFrame({
        "temperature": [temperature],
        "humidity": [humidity],
        "rainfall": [rainfall],
        "soil_moisture": [soil_moisture]
    })
    prediction = irrigation_model.predict(input_data)[0]
    return "Irrigate" if prediction == 1 else "Don't Irrigate"

# Function to predict crop yield
def predict_yield(rainfall, temperature, fertilizer, soil_quality):
    input_data = pd.DataFrame({
        "rainfall": [rainfall],
        "temperature": [temperature],
        "fertilizer": [fertilizer],
        "soil_quality": [soil_quality]
    })
    prediction = yield_model.predict(input_data)[0]
    return round(prediction, 2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/disease", methods=["GET", "POST"])
def disease():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("disease.html", error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("disease.html", error="No file selected")
        if file:
            # Save the uploaded image
            image_path = os.path.join("static", "uploaded_image.jpg")
            file.save(image_path)
            # Predict the disease
            image = Image.open(image_path).convert("RGB")
            predicted_disease = predict_disease(image)
            return render_template("disease.html", prediction=predicted_disease, image_path=image_path)
    return render_template("disease.html")

@app.route("/irrigation", methods=["GET", "POST"])
def irrigation():
    if request.method == "POST":
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])
        soil_moisture = float(request.form["soil_moisture"])
        prediction = predict_irrigation(temperature, humidity, rainfall, soil_moisture)
        return render_template("irrigation.html", prediction=prediction)
    return render_template("irrigation.html")

@app.route("/yield", methods=["GET", "POST"])
def yield_prediction():
    if request.method == "POST":
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])
        fertilizer = float(request.form["fertilizer"])
        soil_quality = float(request.form["soil_quality"])
        prediction = predict_yield(rainfall, temperature, fertilizer, soil_quality)
        return render_template("yield.html", prediction=prediction)
    return render_template("yield.html")

if __name__ == "__main__":
    app.run(debug=True)