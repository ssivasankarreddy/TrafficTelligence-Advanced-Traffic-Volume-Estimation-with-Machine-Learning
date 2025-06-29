from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import datetime

# Initialize Flask app
app = Flask(__name__)

# Load your trained traffic volume model
model = joblib.load('model/traffic_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get values from the form
            datetime_input = request.form['datetime']
            temperature = float(request.form.get('temperature', 20))
            weather_input = request.form['weather']
            event_input = request.form['event']

            # Parse datetime
            dt = pd.to_datetime(datetime_input)
            hour = dt.hour
            dayofweek = dt.dayofweek

            # Encode categorical features
            weather_map = {'Clear': 0, 'Cloudy': 1, 'Rainy': 2, 'Fog': 3}
            event_map = {'No Event': 0, 'Event': 1}
            weather_encoded = weather_map.get(weather_input, 0)
            event_encoded = event_map.get(event_input, 0)

            # Prepare feature vector for prediction
            features = np.array([[hour, dayofweek, temperature, weather_encoded, event_encoded]])
            predicted_volume = model.predict(features)[0]
            prediction = round(predicted_volume, 2)

        except Exception as e:
            print(f"❌ Error: {e}")
            prediction = "Prediction failed. Please check your inputs."

    return render_template('index.html', prediction=prediction, time=datetime.datetime.now())

if __name__ == '__main__':
    app.run(debug=True)




