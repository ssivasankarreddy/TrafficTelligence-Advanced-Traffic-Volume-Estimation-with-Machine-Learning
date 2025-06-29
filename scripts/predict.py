import pandas as pd
import joblib

# Load model and data
model = joblib.load('models/traffic_model.pkl')
df = pd.read_csv('data/processed_data.csv')

# Predict
features = df[['location_encoded', 'hour', 'dayofweek', 'vehicle_count', 'accidents']]
df['predicted_speed'] = model.predict(features)

# Save predictions
df.to_csv('data/predictions.csv', index=False)
print("✅ Predictions saved to data/predictions.csv")
