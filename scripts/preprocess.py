import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    df = pd.read_csv('data/synthetic_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['location_encoded'] = LabelEncoder().fit_transform(df['location'])
    df['weather_encoded'] = LabelEncoder().fit_transform(df['weather'])
    df['event_encoded'] = LabelEncoder().fit_transform(df['event'])

    df.to_csv('data/processed_data.csv', index=False)
    print("✅ Preprocessing with weather & event complete!")

preprocess_data()

