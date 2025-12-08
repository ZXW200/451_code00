import argparse
import sys
import traceback
from pathlib import Path

from utils import create_output_dir, get_device, print_device_info, calculate_batch_size
from feature_extraction import run_feature_extraction
from dimensionality import run_dimensionality_reduction
from cluster import run_clustering_pipeline
from classification import run_classification_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Image Analysis - Feature Extraction, Clustering & Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['resnet50', 'densenet121', 'dinov2'],
        default=['resnet50'],
        help='Models for feature extraction (e.g., --models resnet50 densenet121 dinov2)'
    )
    
    args = parser.parse_args()
    
    try:
        output_dir = create_output_dir('pipeline', 'history')
        device = get_device()
        batch_size = calculate_batch_size(device)
        
        dataset_path = args.dataset
        dataset_name = Path(args.dataset).name
        models = args.models
        
        print(f"Dataset: {dataset_path}")
        print(f"Dataset Name: {dataset_name}")
        print(f"Models: {', '.join(models)}")
        print_device_info(device)
        print(f"Output: {output_dir}\n")
        
        for subdir in ['figures', 'results']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        for model_idx, model_name in enumerate(models, 1):
            print(f"\n{'='*80}")
            print(f"Model {model_idx}/{len(models)}: {model_name}")
            print(f"{'='*80}\n")
            
            print(f"Stage 1: Feature Extraction ({model_name})")
            
            features, labels, metadata = run_feature_extraction(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                output_dir=output_dir
            )
            
            class_names = metadata['class_names']
            
            model_subdir = output_dir / 'results' / model_name
            model_figures_dir = output_dir / 'figures' / model_name
            model_subdir.mkdir(parents=True, exist_ok=True)
            model_figures_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nStage 2: Dimensionality Reduction & Clustering ({model_name})")
            
            model_temp_dir = output_dir / '_temp' / model_name
            (model_temp_dir / 'figures').mkdir(parents=True, exist_ok=True)
            (model_temp_dir / 'results').mkdir(parents=True, exist_ok=True)
            
            dim_results = run_dimensionality_reduction(
                features, labels, class_names, model_temp_dir
            )
            
            features_reduced = dim_results.get('reduced_features', None)
            if features_reduced is not None:
                run_clustering_pipeline(
                    features_reduced, labels, class_names, model_temp_dir
                )
            
            print(f"\nStage 3: Classification ({model_name})")
            
            methods = ['linear', 'knn']
            run_classification_pipeline(
                features, labels, class_names, device, model_temp_dir, methods=methods
            )
            
            import shutil
            if (model_temp_dir / 'results').exists():
                for json_file in (model_temp_dir / 'results').glob('*.json'):
                    shutil.copy(json_file, model_subdir / json_file.name)
            
            if (model_temp_dir / 'figures').exists():
                for fig_file in (model_temp_dir / 'figures').glob('*.png'):
                    shutil.copy(fig_file, model_figures_dir / fig_file.name)
            
            shutil.rmtree(model_temp_dir)
            
            print(f"\nModel {model_name} completed")
        
        temp_parent = output_dir / '_temp'
        if temp_parent.exists():
            import shutil
            shutil.rmtree(temp_parent)
        
        print(f"\n{'='*80}")
        print(f"All models completed - {output_dir}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
