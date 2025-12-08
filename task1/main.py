# main.py
import sys
import traceback
from utils import create_output_dir
from preprocessing import run_preprocessing_task
from cluster import run_clustering_task
import os


# CSV file path - update this to point to your data file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ClimateDataBasel.csv")


def main():
    try:
        output_dir = create_output_dir('pipeline', 'history')
        
        print(f"Data: {DATA_PATH}")
        print(f"Output: {output_dir}\n")
        
        preprocess_dir = output_dir / 'preprocess'
        cluster_dir = output_dir / 'cluster'
        
        for subdir in [preprocess_dir, cluster_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / 'figures').mkdir(exist_ok=True)
        
        print("Preprocessing")
        run_preprocessing_task(DATA_PATH, preprocess_dir)
        
        print("\nClustering")
        run_clustering_task(str(preprocess_dir), cluster_dir)
        
        print(f"\nResults: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

