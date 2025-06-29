import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/processed_data.csv')

# ✅ Updated to use 7 features
X = df[['location_encoded', 'hour', 'dayofweek', 'vehicle_count', 'accidents', 'weather_encoded', 'event_encoded']]
y = df['avg_speed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'model/traffic_model.pkl')
print("✅ Model trained and saved with 7 features!")


