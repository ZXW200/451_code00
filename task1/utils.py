# Utility functions for file IO and data handling
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


# Create a timestamped directory for storing pipeline results
def make_dir(name, base="history"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / f"run_{name}_{ts}"
    path.mkdir(parents=True, exist_ok=True)

    return path


# Load CSV data file and print basic information
def load_data(path):
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

    return df


# Save matplotlib figure to figures subdirectory
def save_fig(fig, path, name):
    p = path / "figures" / f"{name}.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Fig saved: {name}.png")


# Save dataframe to CSV file
def save_csv(df, path, name):
    p = path / name
    df.to_csv(p, index=False)
    print(f"CSV saved: {name}")


# Save dictionary or object to JSON file with proper encoding
def save_json(res, path, name):
    p = path / name
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"JSON saved: {name}")


# Get predefined column names and categories for climate data
def get_cols() :
    # List of all climate feature names
    names = [
        'temp_min', 'temp_max', 'temp_mean',
        'humidity_min', 'humidity_max', 'humidity_mean',
        'pressure_min', 'pressure_max', 'pressure_mean',
        'precipitation', 'snowfall', 'sunshine',
        'wind_gust_min', 'wind_gust_max', 'wind_gust_mean',
        'wind_speed_min', 'wind_speed_max', 'wind_speed_mean'
    ]

    # Group features by category
    cats = {
        'temperature': ['temp_min', 'temp_max', 'temp_mean'],
        'humidity': ['humidity_min', 'humidity_max', 'humidity_mean'],
        'pressure': ['pressure_min', 'pressure_max', 'pressure_mean'],
        'precipitation': ['precipitation', 'snowfall'],
        'sunshine': ['sunshine'],
        'wind_gust': ['wind_gust_min', 'wind_gust_max', 'wind_gust_mean'],
        'wind_speed': ['wind_speed_min', 'wind_speed_max', 'wind_speed_mean']
    }

    return names, cats
