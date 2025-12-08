import argparse
import sys
import traceback
from utils import create_output_dir
from preprocessing import run_preprocessing_task
from cluster import run_clustering_task


def main():
    parser = argparse.ArgumentParser(
        description='Basel Climate Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to climate data CSV file'
    )
    
    args = parser.parse_args()
    
    try:
        output_dir = create_output_dir('pipeline', 'history')
        
        print(f"Data: {args.data}")
        print(f"Output: {output_dir}\n")
        
        preprocess_dir = output_dir / 'preprocess'
        cluster_dir = output_dir / 'cluster'
        
        for subdir in [preprocess_dir, cluster_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / 'figures').mkdir(exist_ok=True)
        
        print("Preprocessing")
        run_preprocessing_task(args.data, preprocess_dir)
        
        print("\nClustering")
        run_clustering_task(str(preprocess_dir), cluster_dir)
        
        print(f"\nResults: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
