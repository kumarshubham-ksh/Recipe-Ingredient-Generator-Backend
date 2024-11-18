from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load pre-trained model and vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    raise FileNotFoundError("Model or Vectorizer file not found. Train the model first!")

@app.route('/predict', methods=['POST'])
def predict_ingredients():
    data = request.get_json()
    description = data.get('description', '')

    if not description:
        return jsonify({"error": "No description provided"}), 400

    # Transform the input and predict ingredients
    input_vector = vectorizer.transform([description])
    prediction = model.predict(input_vector)
    ingredients = prediction[0].split(",")  # Assuming model predicts comma-separated ingredients

    return jsonify({"ingredients": ingredients})

if __name__ == '__main__':
    app.run(debug=True)
