import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
# Ensure these files are in the same directory as app.py
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # 1. Collect inputs
        temp = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        
        # FIX: User inputs m/s, model needs km/h
        # 1 m/s = 3.6 km/h
        ws_ms = float(request.form.get('WS'))
        ws_kmh = ws_ms * 3.6 
        
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        
        # 2. Map Categorical Text to Numbers
        # Model expects 1 for Fire, 0 for Not Fire
        classes = 1.0 if request.form.get('Classes') == "Fire" else 0.0
        # Model expects 0 for Bejaia, 1 for Sidi-Bel Abbes
        region = 1.0 if request.form.get('Region') == "Sidi-Bel Abbes" else 0.0

        # 3. Transform and Predict
        # Order must match scaler: [Temp, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]
        input_data = np.array([[temp, rh, ws_kmh, rain, ffmc, dmc, isi, classes, region]])
        scaled_data = standard_scaler.transform(input_data)
        prediction = ridge_model.predict(scaled_data)
        
        result = round(float(prediction[0]), 2)

        # 4. Logic for Danger Level
        if result < 6:
            danger = "Low"
        elif result < 17:
            danger = "Moderate"
        elif result < 30:
            danger = "High"
        else:
            danger = "Extreme"

        return render_template('home.html', 
                               result=result, 
                               danger_level=danger,
                               fire_status=request.form.get('Classes'),
                               region_name=request.form.get('Region'))

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)