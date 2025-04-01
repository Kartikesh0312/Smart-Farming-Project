# Smart Farming Assistant 🌾🚜
# Overview
The Smart Farming Assistant is a Flask-based web application designed to empower farmers with AI-driven tools for sustainable agriculture. It provides actionable insights to optimize crop health, irrigation, and yield through machine learning and deep learning models, making farming smarter and more efficient.

# What It Achieves
Plant Disease Detection: Identifies plant diseases (e.g., Apple___Black_rot) using a ResNet18 model trained on the PlantVillage dataset (92% accuracy across 39 classes). It offers confidence scores (e.g., 95%) and practical recommendations (e.g., "Apply fungicide and prune affected branches") to help farmers act quickly and prevent crop loss.
Irrigation Prediction: Predicts irrigation needs with a Random Forest model (98% accuracy) based on inputs like temperature, humidity, rainfall, and soil moisture, ensuring crops get the right amount of water to thrive.
Crop Yield Estimation: Estimates crop yield (R² score of 0.95) using factors like rainfall, temperature, fertilizer, and soil quality, helping farmers plan harvests and maximize productivity.
User-Friendly Design: Features an intuitive interface built with HTML, CSS, and Jinja2, complete with a thematic cornfield background and a footer for a professional look.

# Benefits for Farmers
This project helps farmers make data-driven decisions, reducing guesswork and improving outcomes. It detects diseases early to save crops, optimizes irrigation to conserve water, and predicts yields to plan better—ultimately boosting productivity and sustainability for small-scale farmers.

# Tech Stack
Backend: Flask, Python
Machine Learning: PyTorch (ResNet18), Scikit-learn (Random Forest)
Frontend: HTML, CSS
