# crop_predict.py

import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------
# Load ML artifacts
# -----------------------
import os
base_path = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(base_path, "models/Crop_Recomm/crop_recommendation_CNN.keras"))
label_encoder = joblib.load(os.path.join(base_path, "models/Crop_Recomm/crop_label_encoder.pkl"))
scaler = joblib.load(os.path.join(base_path, "models/Crop_Recomm/feature_scaler.pkl"))

print("Crop recommendation model loaded successfully")


# -----------------------
# Prediction Function
# -----------------------
def predict_crop(data):

    # Input order
    features = np.array([
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(features)

    # CNN reshape
    cnn_input = scaled_features.reshape(scaled_features.shape[0], scaled_features.shape[1], 1)

    # Predict
    pred_probs = model.predict(cnn_input)
    pred_class = np.argmax(pred_probs, axis=1)

    crop = label_encoder.inverse_transform(pred_class)[0]

    return crop

if __name__ == "__main__":
    print("\n=== Crop Recommendation System ===")
    print("Enter the following soil and climate values:\n")
    data = {
        "N":           float(input("Nitrogen (N) content in soil    : ")),
        "P":           float(input("Phosphorus (P) content in soil  : ")),
        "K":           float(input("Potassium (K) content in soil   : ")),
        "temperature": float(input("Temperature (°C)                : ")),
        "humidity":    float(input("Humidity (%)                    : ")),
        "ph":          float(input("Soil pH                         : ")),
        "rainfall":    float(input("Rainfall (mm)                   : ")),
    }
    prediction = predict_crop(data)
    print(f"\n✅ Recommended Crop: {prediction}")