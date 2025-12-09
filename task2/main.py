# Main pipeline for image feature extraction and analysis
import argparse
import sys
import traceback
import os
import shutil
from pathlib import Path
from utils import make_dir, get_dev, calc_bs,print_dev
from feature_extraction import run_ext
from dimensionality import run_dim_red
from cluster import run_clus
from classification import run_class
# Check if kagglehub is installed for dataset downloads
try:
    import kagglehub
except ImportError:
    print("pip install kagglehub")
    sys.exit(1)



# Datasets to process
DATASETS = ['cats_dogs', 'food101']

# Models to use for each dataset
MODELS = {
    'cats_dogs': ['resnet50', 'densenet121', 'dinov2'],
    'food101': ['resnet50', 'densenet121', 'dinov2']
}

# Dataset information including Kaggle identifiers
INFO = {
    'cats_dogs': {
        'name': 'Cats vs Dogs',
        'kaggle': 'karakaggle/kaggle-cat-vs-dog-dataset'
    },
    'food101': {
        'name': 'Food-101',
        'kaggle': 'dansbecker/food-101'
    }
}


# Create symbolic link to dataset directory
def link_data(src, name):
    local_dir = Path(__file__).parent / 'Dataset'
    local_dir.mkdir(exist_ok=True)
    dst = local_dir / name

    # Check if link already exists
    if dst.exists():
        if dst.is_symlink() or dst.is_dir():
            print(f"Found: {dst}")
            return str(dst)

    # Create symbolic link or return source path if linking fails
    try:
        if dst.exists(): dst.unlink()
        dst.symlink_to(Path(src), target_is_directory=True)
        print(f"Linked: {dst}")
        return str(dst)
    except (OSError, NotImplementedError):
        print(f"Link fail, using: {src}")
        return str(src)


# Download and prepare Cats vs Dogs dataset from Kaggle
def prep_cd():
    h = INFO['cats_dogs']['kaggle']
    print(f"\nDL CatsDogs: {h}")
    try:
        # Download dataset using kagglehub
        p = kagglehub.dataset_download(h)
        print(f"Cached: {p}")

        # Find directory containing cat and dog subdirectories
        src = None
        for root, dirs, files in os.walk(p):
            sub = [d.lower() for d in dirs]
            if 'cat' in sub and 'dog' in sub:
                src = str(Path(root))
                break
        if not src: src = str(p)
        return link_data(src, 'cats_dogs')

    except Exception as e:
        print(f"Err: {e}")
        raise e


# Download and prepare Food-101 dataset from Kaggle
def prep_food():
    h = INFO['food101']['kaggle']
    print(f"\nDL Food101: {h}")
    try:
        # Download dataset using kagglehub
        p = kagglehub.dataset_download(h)
        print(f"Cached: {p}")

        # Find images directory or food class directories
        src = None
        for root, dirs, files in os.walk(p):
            if 'images' in dirs:
                src = str(Path(root) / 'images')
                break
        if not src:
            for root, dirs, files in os.walk(p):
                if 'apple_pie' in dirs:
                    src = str(root)
                    break
        if not src: src = str(p)
        return link_data(src, 'food101')

    except Exception as e:
        print(f"Err: {e}")
        raise e


# Main pipeline execution
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    try:
        # Setup output directory and compute device
        out = make_dir('pipeline', 'history')
        dev = get_dev(args.device)
        bs = calc_bs(dev)

        print(f"Out: {out}")
        print_dev(dev)

        # Create output subdirectories
        for d in ['figures', 'results']:
            (out / d).mkdir(parents=True, exist_ok=True)

        # Process each dataset
        for ds in DATASETS:
            print(f"\nDataset: {ds}")

            # Select models to run
            run_list = args.models if args.models else MODELS.get(ds, [])
            print(f"Models: {run_list}")

            # Download and prepare dataset
            if ds == 'cats_dogs':
                d_path = prep_cd()
            elif ds == 'food101':
                d_path = prep_food()
            else:
                continue

            # Run pipeline for each model
            for i, m_name in enumerate(run_list, 1):
                print(f"\n--- {m_name} ({i}/{len(run_list)}) ---")

                # Step 1: Extract features using pretrained model
                feats, y, meta = run_ext(
                    d_path, ds, m_name, dev, bs, out
                )

                # Create temporary directory for this model run
                names = meta['classes']
                tid = f"{ds}_{m_name}"
                tmp = out / '_temp' / tid
                (tmp / 'figures').mkdir(parents=True, exist_ok=True)
                (tmp / 'results').mkdir(parents=True, exist_ok=True)

                # Step 2: Dimensionality reduction with UMAP
                dim_res = run_dim_red(feats, y, names, tmp)

                # Step 3: Clustering on reduced features
                if 'reduced' in dim_res:
                    _ = run_clus(
                        dim_res['reduced'], y, names, tmp, fix_k=None
                    )

                # Step 4: Classification experiments
                run_class(feats, y, names, dev, tmp, methods=['linear', 'knn'])

                # Move results to final output location
                res_d = out / 'results' / ds / m_name
                fig_d = out / 'figures' / ds / m_name
                res_d.mkdir(parents=True, exist_ok=True)
                fig_d.mkdir(parents=True, exist_ok=True)

                # Copy all results and figures
                for f in (tmp / 'results').glob('*'):
                    shutil.copy(f, res_d / f.name)
                for f in (tmp / 'figures').glob('*'):
                    shutil.copy(f, fig_d / f.name)

        # Cleanup temporary directory
        if (out / '_temp').exists():
            shutil.rmtree(out / '_temp')

        print(f"\nDone. Saved to: {out}")

    except Exception as e:
        print(f"\nERR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# Entry point
if __name__ == '__main__':
    main()
