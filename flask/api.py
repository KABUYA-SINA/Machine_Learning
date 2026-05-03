from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load model function
def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features"]


# Safe model loading
try:
    model, feature_names = load_model()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    model = None


@app.route("/")
def home():
    logging.info("Home endpoint called")
    return {
        "message": "ML API is running",
        "endpoints": ["/predict", "/visualization", "/health"]
    }


@app.route("/health")
def health():
    logging.info("Health check requested")
    return {"status": "ok"}


@app.route("/visualization")
def visualization():
    logging.info("Visualization requested")

    # Load dataset
    df = pd.read_csv("data/dataset.csv")

    # Check required column
    if "PINCP" not in df.columns:
        return jsonify({"error": "Column PINCP not found"}), 400

    # Create plot
    plt.figure()
    df["PINCP"].hist()

    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return f'<img src="data:image/png;base64,{plot_url}"/>'


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Prediction request received")

    # Check model availability
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    try:
        # Prepare input
        features = np.array(data["features"]).reshape(1, -1)

        # Prediction
        prediction = model.predict(features)[0]

        # probability
        proba = model.predict_proba(features)[0]

        prob_class_0 = proba[0]
        prob_class_1 = proba[1]

        logging.info("Prediction successful")

        return jsonify({
            "prediction": int(prediction),
             "probabilities": {
                "0": float(prob_class_0),
                "1": float(prob_class_1)
            }
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)