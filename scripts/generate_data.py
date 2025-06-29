import pandas as pd
import numpy as np

def simulate_traffic_data(rows=1000):
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=rows, freq='h'),
        'location': np.random.choice(['North', 'South', 'East', 'West'], rows),
        'vehicle_count': np.random.randint(10, 100, rows),
        'avg_speed': np.random.uniform(20, 80, rows),
        'accidents': np.random.poisson(0.2, rows),
        'weather': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Fog'], rows),
        'event': np.random.choice(['Yes', 'No'], rows)
    }
    df = pd.DataFrame(data)
    df.to_csv('data/synthetic_data.csv', index=False)
    print("✅ Data with weather & event generated!")

simulate_traffic_data()


