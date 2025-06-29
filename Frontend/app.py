from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

app = Flask(__name__)

model = tf.keras.models.load_model('my_model.keras')

iris = load_iris()
scaler = StandardScaler()
scaler.fit(iris.data)

class_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return "🌸 Iris Classifier API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or len(features) != 4:
            return jsonify({"error": "Input must be a list of 4 numeric features"}), 400

        input_data = np.array([features], dtype=np.float32)
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled, verbose=0)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/performance', methods=['GET'])
def performance():
    return jsonify({
        "test_accuracy": 0.9000,
        "test_loss": 0.2024,
        "total_parameters": 2979,
        "per_class_accuracy": {
            "setosa": 1.0000,
            "versicolor": 0.8000,
            "virginica": 0.9000
        },
        "model_complexity": {
            "num_layers": 6,
            "hidden_layers": 5,
            "total_parameters": 2979
        },
        "training_efficiency": {
            "epochs_trained": 50,
            "training_accuracy": 0.9271,
            "validation_accuracy": 0.9583
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)