# predict_fertilizer.py — Fertilizer Recommendation

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "fertilizer")

# ── Load models at import time ─────────────────────────────────────────
fertilizer_model = None
le_soil = None
le_crop = None
le_fertilizer = None

try:
    fertilizer_model = joblib.load(os.path.join(MODEL_DIR, "fertilizer_prediction_model.joblib"))
    le_soil = joblib.load(os.path.join(MODEL_DIR, "soil_type_encoder.joblib"))
    le_crop = joblib.load(os.path.join(MODEL_DIR, "crop_type_encoder.joblib"))
    le_fertilizer = joblib.load(os.path.join(MODEL_DIR, "fertilizer_name_encoder.joblib"))
    print("✅ Fertilizer prediction model loaded")
except Exception as e:
    print(f"⚠️ Fertilizer model failed to load: {e}")


# ── Main prediction function ───────────────────────────────────────────
def predict_fertilizer(data):
    """
    Predict optimal fertilizer.
    data dict keys: temperature, humidity, moisture, soil_type, crop_type,
                    nitrogen, potassium, phosphorous
    Returns dict with predicted_fertilizer.
    """
    if fertilizer_model is None:
        raise RuntimeError("Fertilizer model is not loaded")

    input_data = {
        "Temparature": [float(data["temperature"])],
        "Humidity ": [float(data["humidity"])],
        "Moisture": [int(data["moisture"])],
        "Soil Type": int(le_soil.transform([data["soil_type"]])[0]),
        "Crop Type": int(le_crop.transform([data["crop_type"]])[0]),
        "Nitrogen": [int(data["nitrogen"])],
        "Potassium": [int(data["potassium"])],
        "Phosphorous": [int(data["phosphorous"])],
    }
    input_df = pd.DataFrame(input_data)

    prediction = fertilizer_model.predict(input_df)
    predicted_fertilizer = str(le_fertilizer.inverse_transform(prediction)[0])

    return {"predicted_fertilizer": predicted_fertilizer}
