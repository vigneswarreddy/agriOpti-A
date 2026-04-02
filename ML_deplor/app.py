# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from crop_predict import predict_crop
from predict_disease import predict_plant_disease
from predict_yield import predict_crop_yield
from predict_land import predict_aerial_land
from predict_fertilizer import predict_fertilizer

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the React frontend


# -----------------------
# Health Check
# -----------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Crop Recommendation API Running"
    })


# -----------------------
# Prediction Endpoint
# -----------------------
@app.route("/predict-crop", methods=["POST"])
def predict():

    data = request.get_json()

    try:
        crop = predict_crop(data)

        return jsonify({
            "recommended_crop": crop
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# -----------------------
# Plant Disease Endpoint
# -----------------------
@app.route("/predict-disease", methods=["POST"])
def disease_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Read the file bytes
        image_bytes = file.read()
        
        # Get prediction
        result = predict_plant_disease(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------
# Crop Yield Endpoint
# -----------------------
@app.route("/predict-yield", methods=["POST", "OPTIONS"])
def yield_predict():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response, 200

    data = request.get_json()
    print(f"\n[INFO] --- INCOMING /predict-yield REQUEST ---")
    print(f"[INFO] Payload Data: {data}")

    try:
        result = predict_crop_yield(data)
        print(f"[INFO] Prediction Result Generated: {result}")
        print(f"[INFO] --- END /predict-yield REQUEST ---\n")
        response = jsonify(result)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except Exception as e:
        print(f"[ERROR] Prediction Failed: {e}")
        print(f"[INFO] --- END /predict-yield REQUEST ---\n")
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 400


# -----------------------
# Aerial Land Analysis Endpoint
# -----------------------
@app.route("/predict-land", methods=["POST"])
def land_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    lat = request.form.get('lat', 28.6, type=float)
    lon = request.form.get('lon', 77.2, type=float)

    try:
        image_bytes = file.read()
        result = predict_aerial_land(image_bytes, lat, lon)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Land analysis failed: {e}")
        return jsonify({"error": str(e)}), 400


# -----------------------
# Fertilizer Prediction Endpoint
# -----------------------
@app.route("/predict-fertilizer", methods=["POST"])
def fertilizer_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ['temperature', 'humidity', 'moisture', 'soil_type', 'crop_type', 'nitrogen', 'potassium', 'phosphorous']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        result = predict_fertilizer(data)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Fertilizer prediction failed: {e}")
        return jsonify({"error": str(e)}), 400


# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)