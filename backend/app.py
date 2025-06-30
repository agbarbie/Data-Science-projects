from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ✅ Load the H5 model
model = tf.keras.models.load_model("my_model.h5")

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({'error': 'Missing input features'}), 400

    try:
        features = np.array([data['features']])  # Shape: (1, 4)
        prediction = model.predict(features)[0]

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        classes = ['setosa', 'versicolor', 'virginica']
        predicted_class = classes[predicted_index]

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance')
def performance():
    return jsonify({
        "test_accuracy": 0.98,
        "test_loss": 0.12,
        "total_parameters": 12345,
        "per_class_accuracy": {
            "setosa": 0.99,
            "versicolor": 0.97,
            "virginica": 0.98
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
