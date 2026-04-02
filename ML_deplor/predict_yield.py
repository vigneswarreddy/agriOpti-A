import joblib
import pandas as pd
import os

# Load the trained crop yield model and columns
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'Crop_yelid')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_crop_yield_model.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.pkl')

try:
    yield_model = joblib.load(MODEL_PATH)
    yield_columns = joblib.load(COLUMNS_PATH)
    print("Crop Yield Model and Columns loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load Crop Yield Model or Columns. Error: {e}")
    yield_model = None
    yield_columns = None

def predict_crop_yield(data):
    try:
        # Extract features
        rainfall = float(data.get('rainfall', 0))
        temperature = float(data.get('temperature', 0))
        soil_ph = float(data.get('soil_ph', 0))
        fertilizer_used = float(data.get('fertilizer_used', 0))
        crop_type = data.get('crop_type', '')
        irrigation_type = data.get('irrigation_type', '')

        input_data = pd.DataFrame({
            'rainfall': [rainfall],
            'temperature': [temperature],
            'soil_ph': [soil_ph],
            'fertilizer_used': [fertilizer_used]
        })

        # Crop types (wheat, rice, maize, soyabean) - adjusting strictly to model strings if necessary
        # The model might expect 'Wheat', 'Rice', 'Maize', 'Soybean' based on earlier checks
        for crop in ['Wheat', 'Rice', 'Maize', 'Soybean']:
            # Mappings for user inputs to model features based on previous knowledge
            input_val = 1 if crop.lower() == crop_type.lower() or (crop == 'Soybean' and crop_type.lower() == 'soyabean') else 0
            input_data[f'crop_type_{crop}'] = input_val

        # Irrigation types
        # The prompt says: drip, flood water, Sprinkler Irrigation
        irrigations = ['Sprinkler Irrigation', 'Flood Irrigation', 'Drip Irrigation']
        for irrigation in irrigations:
            col_name = f'irrigation_type_{irrigation}'
            # matching 'flood water' to 'Flood Irrigation', 'drip' to 'Drip Irrigation'
            user_irr_mapped = irrigation_type.lower()
            if 'flood' in user_irr_mapped:
                user_irr_mapped = 'flood irrigation'
            elif 'drip' in user_irr_mapped:
                user_irr_mapped = 'drip irrigation'
            elif 'sprinkler' in user_irr_mapped:
                user_irr_mapped = 'sprinkler irrigation'

            if col_name in input_data.columns or (yield_columns is not None and col_name in yield_columns):
                input_data[col_name] = 1 if irrigation.lower() == user_irr_mapped else 0

        # Add missing columns
        if yield_columns is not None:
            missing_cols = set(yield_columns) - set(input_data.columns)
            for col in missing_cols:
                input_data[col] = 0

            # Reorder columns
            input_data = input_data[yield_columns]

        # Predict
        if yield_model is None:
            raise Exception("Model is not loaded.")
            
        prediction = yield_model.predict(input_data)[0]
        
        # Ensure it's not negative
        predicted_yield = max(0, float(prediction))
        
        return {
            "predicted_yield": round(predicted_yield, 2),
            "unit": "Quintals per Hectare"
        }
        
    except Exception as e:
        raise Exception(f"Crop yield prediction failed: {str(e)}")
