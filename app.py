import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize the Flask App
app = Flask(__name__)
if __name__ == "__main__":
    app.run()


# --- 1. MODEL LOADING LOGIC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'glof_model.pkl')

model = None

print("\n" + "="*50)
print("STEP 1: INITIALIZING SYSTEM...")

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print("✅ SUCCESS: Model brain loaded into memory!")
    except Exception as e:
        print(f"❌ ERROR: Model file found but failed to load: {e}")
else:
    print(f"❌ ERROR: 'glof_model.pkl' NOT FOUND in {BASE_DIR}")

print("STEP 2: STARTING FLASK SERVER...")
print("="*50 + "\n")

# --- 2. ROUTES ---

@app.route('/')
def home():
    """Renders the HTML interface from the templates folder."""
    return {'message': 'GLOF Risk Prediction API is running.'}

@app.route('/predict', methods=['POST'])
def predict():
    """Receives data, predicts risk, and returns JSON."""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded.'}), 500

    try:
        data = request.get_json()
        
        # Format the data into a DataFrame for XGBoost
        # Features: Lake_Area_km2, Dam_Slope_deg, Lake_Temp_C, Elevation_m
        input_data = pd.DataFrame([{
            'Lake_Area_km2': float(data['Lake_Area_km2']),
            'Dam_Slope_deg': float(data['Dam_Slope_deg']),
            'Lake_Temp_C': float(data['Lake_Temp_C']),
            'Elevation_m': float(data['Elevation_m'])
        }])
        
        # Predict
        raw_score = float(model.predict(input_data)[0])
        risk_score = max(0, min(1, raw_score)) # Keep between 0-1

        # Logic for Risk Levels
        if risk_score <= 0.30:
            lvl, clr, adv = "LOW", "#2ecc71", "Normal monitoring."
        elif risk_score <= 0.60:
            lvl, clr, adv = "MODERATE", "#f1c40f", "Increase observation frequency."
        elif risk_score <= 0.80:
            lvl, clr, adv = "HIGH", "#e67e22", "Warning: Structural survey recommended."
        else:
            lvl, clr, adv = "IMMEDIATE ACTION", "#e74c3c", "Danger: Evacuation protocol may be needed."

        return jsonify({
            'status': 'success',
            'risk_index': round(risk_score, 4),
            'risk_level': lvl,
            'recommendation': adv,
            'color': clr
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- 3. RUN SERVER ---
# This block MUST be at the very bottom and NOT indented
if __name__ == '__main__':
    print("🚀 Server is launching now...")
    # Using 0.0.0.0 makes it accessible on your local network
    app.run(host='127.0.0.1', port=5000, debug=False)