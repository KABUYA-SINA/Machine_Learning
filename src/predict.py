import pickle
import numpy as np


def load_model(path="models/model.pkl"):
    # Load trained model from disk
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at: {path}")


def predict(model, data):
    # Convert input data to numpy array for consistency
    data = np.array(data)

    # Ensure input has correct shape (2D array required for sklearn)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    # Make prediction using trained model
    return model.predict(data)