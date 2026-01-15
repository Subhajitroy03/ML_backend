# predict.py
import sys
import pickle
import pandas as pd
import json
import os

# Hide warnings to keep console output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main():
    try:
        # Load the model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, 'glof_model.pkl'), 'rb') as f:
            model = pickle.load(f)

        # Get data from command line arguments (sent by Express)
        # Format: area slope temp elev
        area = float(sys.argv[1])
        slope = float(sys.argv[2])
        temp = float(sys.argv[3])
        elev = float(sys.argv[4])

        # Prepare features
        features = pd.DataFrame([[area, slope, temp, elev]], 
                                columns=['Lake_Area_km2', 'Dam_Slope_deg', 'Lake_Temp_C', 'Elevation_m'])

        # Predict
        prediction = float(model.predict(features)[0])
        # Clamp to 0-1
        risk_score = max(0.0, min(1.0, prediction))

        # Map numeric score to level and advice
        if risk_score <= 0.30:
            lvl, adv = "LOW", "Normal monitoring."
        elif risk_score <= 0.60:
            lvl, adv = "MODERATE", "Increase observation frequency."
        elif risk_score <= 0.80:
            lvl, adv = "HIGH", "Warning: Structural survey recommended."
        else:
            lvl, adv = "IMMEDIATE ACTION", "Danger: Evacuation protocol may be needed."

        # Output result as JSON so callers can parse it
        print(json.dumps({
            "risk_index": round(risk_score, 4),
            "risk_level": lvl,
            "recommendation": adv
        }))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()