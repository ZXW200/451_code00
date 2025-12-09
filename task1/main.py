import sys
import traceback
from utils import make_dir
from preprocessing import run_prep
from cluster import run_cluster


# Path to climate data file
DATA_FILE = "ClimateDataBasel.csv"


def main():
    try:
        # Create output directory with timestamp
        out_dir = make_dir('pipeline', 'history')

        print(f"Data: {DATA_FILE}")
        print(f"Out: {out_dir}\n")

        # Set up subdirectories for preprocessing and clustering
        p_dir = out_dir / 'prep'
        c_dir = out_dir / 'cluster'

        # Create required directory structure
        for d in [p_dir, c_dir]:
            d.mkdir(parents=True, exist_ok=True)
            (d / 'figures').mkdir(exist_ok=True)

        # Run data preprocessing step
        print("Preprocessing")
        run_prep(DATA_FILE, p_dir)

        # Run clustering analysis step
        print("\nClustering")
        run_cluster(str(p_dir), c_dir)

        print(f"\nResults: {out_dir}")

    except Exception as e:
        # Handle and print any errors that occur
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# Execute main function when script is run directly
if __name__ == '__main__':
    main()

