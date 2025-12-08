import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def create_output_dir(task_name: str, base_dir: str = "history") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"instance_{task_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    
    return output_dir




def load_climate_data(file_path: str) -> pd.DataFrame:
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Records: {len(df)}, Features: {len(df.columns)}")
    
    return df


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    figure_path = output_dir / "figures" / f"{filename}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}.png")


def save_dataframe(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    file_path = output_dir / filename
    df.to_csv(file_path, index=False)
    print(f"Saved: {filename}")


def save_results(results: dict, output_dir: Path, filename: str) -> None:
    file_path = output_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {filename}")


def get_feature_names() -> Tuple[list, list]:
    feature_names = [
        'temp_min', 'temp_max', 'temp_mean',
        'humidity_min', 'humidity_max', 'humidity_mean',
        'pressure_min', 'pressure_max', 'pressure_mean',
        'precipitation', 'snowfall', 'sunshine',
        'wind_gust_min', 'wind_gust_max', 'wind_gust_mean',
        'wind_speed_min', 'wind_speed_max', 'wind_speed_mean'
    ]
    
    feature_categories = {
        'temperature': ['temp_min', 'temp_max', 'temp_mean'],
        'humidity': ['humidity_min', 'humidity_max', 'humidity_mean'],
        'pressure': ['pressure_min', 'pressure_max', 'pressure_mean'],
        'precipitation': ['precipitation', 'snowfall'],
        'sunshine': ['sunshine'],
        'wind_gust': ['wind_gust_min', 'wind_gust_max', 'wind_gust_mean'],
        'wind_speed': ['wind_speed_min', 'wind_speed_max', 'wind_speed_mean']
    }
    
    return feature_names, feature_categories
