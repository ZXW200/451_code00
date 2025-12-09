import sys
import traceback
from utils import make_dir
from preprocessing import run_prep
from cluster import run_cluster


DATA_FILE = "ClimateDataBasel.csv"


def main():
    try:
        out_dir = make_dir('pipeline', 'history')

        print(f"Data: {DATA_FILE}")
        print(f"Out: {out_dir}\n")

        p_dir = out_dir / 'prep'
        c_dir = out_dir / 'cluster'

        for d in [p_dir, c_dir]:
            d.mkdir(parents=True, exist_ok=True)
            (d / 'figures').mkdir(exist_ok=True)

        print("Preprocessing")
        run_prep(DATA_FILE, p_dir)

        print("\nClustering")
        run_cluster(str(p_dir), c_dir)

        print(f"\nResults: {out_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
