from flask import Flask, request, jsonify # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import tensorflow_hub as hub # type: ignore
import tensorflow as tf # type: ignore
import numpy as np
import re

# === Initialize Flask App ===
app = Flask(__name__)

# === Load Model and USE Embedder ===
print("Loading model and USE...")
model = load_model('mental_health_classifier.h5', custom_objects={'KerasLayer': hub.KerasLayer})
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("✅ Model and USE loaded!")

# === Preprocessing Function (from clean_dataset.py) ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"u/\w+", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\b(edit|tl;dr|tldr|update):?.*", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

# === Labels Order ===
labels = ["Depression", "Anxiety", "Stress", "PTSD", "ADHD", "OCD", "Schizophrenia"]

# === Predict Route ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({'error': 'No text provided'}), 400

        # Clean and preprocess
        cleaned_text = clean_text(input_text)

        # Embed using USE
        embedded_text = embed([cleaned_text])

        # Predict
        prediction = model.predict(embedded_text).flatten()

        # Get top label and confidence
        label_idx = np.argmax(prediction)
        predicted_label = labels[label_idx]
        confidence = float(prediction[label_idx])

        # Assign risk based on confidence threshold
        if confidence > 0.8:
            risk = 5
        elif confidence > 0.6:
            risk = 4
        elif confidence > 0.4:
            risk = 3
        elif confidence > 0.2:
            risk = 2
        else:
            risk = 1

        return jsonify({
            'label': predicted_label,
            'confidence': confidence,
            'risk': risk
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Home Test Route (optional) ===
@app.route('/', methods=['GET'])
def home():
    return "Mental Health Classifier API is running."

# === Run Server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)