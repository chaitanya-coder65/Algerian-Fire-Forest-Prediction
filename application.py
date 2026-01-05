from flask import Flask, request, render_template
import numpy as np
import pickle

application = Flask(__name__)
app = application

# Load models (Ensure the 'models/' folder exists or remove the prefix if files are in the main folder)
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except:
    # Fallback if files are in the root directory
    ridge_model = pickle.load(open('ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    # Renders the Landing Page
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'POST':
        # 1. Get data from form - names must match HTML 'name' attribute exactly
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # 2. Transform data
        new_scaled_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # 3. Predict
        result = ridge_model.predict(new_scaled_data)

        # 4. Logic for Safety Status (Using FWI > 6 as danger threshold)
        if result[0] > 6:
            forest_status = "The Forest is in Danger!"
        else:
            forest_status = "The Forest is Safe."

        # IMPORTANT: Return to home.html to show results on the form page
        return render_template('home.html', result=round(result[0], 2), forest_status=forest_status)
    
    else:
        # If it's a GET request, just show the blank form
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)