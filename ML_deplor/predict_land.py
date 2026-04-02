# predict_land.py — Sentinel-2 Aerial Land Analysis

import os
import torch
import torch.nn as nn
import pickle
import cv2
import numpy as np
import requests
from google import genai

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths relative to ML/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentinel2", "best_improved_crop_classifier.pth")
SCALER_PATH = os.path.join(BASE_DIR, "models", "sentinel2", "feature_scaler.pkl")

WEATHER_API_KEY = "585c16e678fc74a01145af155437ec10"
GEMINI_API_KEY = "AIzaSyDEC0IBLKaFLr0Znkbq_4kQPdWyuPWlkuo"

# Initialize Gemini
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ── Model definition ──────────────────────────────────────────────────
class ImprovedCropClassificationCNN(nn.Module):
    def __init__(self, input_features=13, num_classes=4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features


# ── Load models at import time ─────────────────────────────────────────
sentinel_model = None
scaler = None

try:
    sentinel_model = ImprovedCropClassificationCNN(input_features=13, num_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    sentinel_model.load_state_dict(checkpoint["model_state_dict"])
    sentinel_model.eval()
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("✅ Sentinel-2 model loaded")
except Exception as e:
    print(f"⚠️ Sentinel-2 model failed to load: {e}")

SELECTED_CLASSES = {0: "wheat", 1: "rye", 2: "barley", 3: "forage_crops"}


# ── Helper functions ────────────────────────────────────────────────────
def _extract_bands_from_image(image_bytes):
    """Simulate Sentinel B4, B5, B8 bands from RGB image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    B4 = np.mean(rgb[:, :, 0]) / 255.0
    B5 = np.mean(rgb[:, :, 1]) / 255.0
    B8 = np.mean(rgb[:, :, 2]) / 255.0
    return B4, B5, B8


def _engineer_features(B4, B5, B8):
    NDVI = (B8 - B4) / (B8 + B4 + 1e-8)
    NDRE = (B8 - B5) / (B8 + B5 + 1e-8)
    SR = B8 / (B4 + 1e-8)
    RE_ratio = B5 / (B4 + 1e-8)
    diff1 = B8 - B4
    diff2 = B8 - B5
    diff3 = B5 - B4
    return np.array([B4, B5, B8, NDVI, NDRE, SR, RE_ratio, diff1, diff2, diff3, B4, B5, B8])


def _get_weather(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=10).json()
        return {
            "temperature": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "condition": res["weather"][0]["description"],
        }
    except Exception:
        return {"temperature": 25, "humidity": 60, "condition": "unavailable"}


# ── Main prediction function ───────────────────────────────────────────
def predict_aerial_land(image_bytes, lat=28.6, lon=77.2):
    """
    Analyse an aerial/satellite image.
    Returns dict with predicted_crop, confidence, ndvi, ndre, weather, farming_strategy.
    """
    if sentinel_model is None or scaler is None:
        raise RuntimeError("Sentinel-2 model is not loaded")

    B4, B5, B8 = _extract_bands_from_image(image_bytes)
    features = _engineer_features(B4, B5, B8)
    scaled = scaler.transform(features.reshape(1, -1))
    tensor = torch.FloatTensor(scaled).to(DEVICE)

    with torch.no_grad():
        outputs, _ = sentinel_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    crop = SELECTED_CLASSES[pred.item()]
    confidence = conf.item()

    ndvi = float((B8 - B4) / (B8 + B4 + 1e-8))
    ndre = float((B8 - B5) / (B8 + B5 + 1e-8))
    weather = _get_weather(lat, lon)

    # Gemini farming strategy
    prompt = (
        f"Satellite image analysis: NDVI={ndvi:.2f}, NDRE={ndre:.2f}. "
        f"Detected crop type: {crop} (confidence={confidence*100:.1f}%). "
        f"Weather: temperature {weather['temperature']}°C, humidity {weather['humidity']}%, "
        f"condition: {weather['condition']}. "
        f"Give a 3-line farming strategy and health interpretation."
    )
    try:
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
        strategy = response.text
    except Exception as e:
        strategy = f"Could not generate strategy: {e}"

    return {
        "predicted_crop": crop,
        "confidence": round(confidence * 100, 2),
        "ndvi": round(ndvi, 3),
        "ndre": round(ndre, 3),
        "weather": weather,
        "farming_strategy": strategy,
    }
