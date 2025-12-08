import argparse
import sys
import traceback
import os
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm

try:
    import kagglehub
except ImportError:
    print("请先安装 kagglehub: pip install kagglehub")
    sys.exit(1)

from utils import create_output_dir, get_device, print_device_info, calculate_batch_size
from feature_extraction import run_feature_extraction
from dimensionality import run_dimensionality_reduction
from cluster import run_clustering_pipeline
from classification import run_classification_pipeline

# ==========================================
# 核心配置区域
# ==========================================

TARGET_DATASETS = ['cats_dogs', 'food101']

DATASET_MODELS_CONFIG = {
    'cats_dogs': ['resnet50', 'densenet121', 'dinov2'],
    'food101': ['resnet50', 'densenet121', 'dinov2']
}

DATASET_INFO = {
    'cats_dogs': {
        'name': 'Cats vs Dogs',
        'kaggle_handle': 'karakaggle/kaggle-cat-vs-dog-dataset'
    },
    'food101': {
        'name': 'Food-101',
        'kaggle_handle': 'dansbecker/food-101'
    }
}


# ==========================================

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def prepare_cats_dogs() -> str:
    handle = DATASET_INFO['cats_dogs']['kaggle_handle']
    print(f"\n[Cats vs Dogs] 正在通过 kagglehub 获取 {handle} ...")
    try:
        path = kagglehub.dataset_download(handle)
        for root, dirs, files in os.walk(path):
            lower_dirs = [d.lower() for d in dirs]
            if 'cat' in lower_dirs and 'dog' in lower_dirs:
                return str(Path(root))
        return str(path)
    except Exception as e:
        print(f"Cats vs Dogs 下载失败: {e}")
        raise e


def prepare_food101() -> str:
    handle = DATASET_INFO['food101']['kaggle_handle']
    print(f"\n[Food-101] 正在通过 kagglehub 获取 {handle} ...")
    try:
        path = kagglehub.dataset_download(handle)
        for root, dirs, files in os.walk(path):
            if 'images' in dirs:
                return str(Path(root) / 'images')
        for root, dirs, files in os.walk(path):
            if 'apple_pie' in dirs:
                return str(root)
        return str(path)
    except Exception as e:
        print(f"Food-101 下载失败: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', type=str, nargs='+', default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    try:
        output_dir = create_output_dir('pipeline', 'history')
        device = get_device(args.device)
        batch_size = calculate_batch_size(device)

        print(f"输出目录: {output_dir}")
        print(f"计算设备: {device}")

        for subdir in ['figures', 'results']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        for dataset_name in TARGET_DATASETS:
            print(f"\n{'=' * 60}")
            print(f"准备处理数据集: {dataset_name}")
            print(f"{'=' * 60}")

            if args.models:
                current_models = args.models
            else:
                current_models = DATASET_MODELS_CONFIG.get(dataset_name, [])

            print(f"计划运行模型: {current_models}")

            if dataset_name == 'cats_dogs':
                dataset_path_str = prepare_cats_dogs()
            elif dataset_name == 'food101':
                dataset_path_str = prepare_food101()
            else:
                continue



            for model_idx, model_name in enumerate(current_models, 1):
                print(f"\n--- 模型 [{model_name}] ({model_idx}/{len(current_models)}) 处理 {dataset_name} ---")

                # Step 1: Feature Extraction
                features, labels, metadata = run_feature_extraction(
                    dataset_path=dataset_path_str,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    output_dir=output_dir
                )

                class_names = metadata['class_names']
                task_id = f"{dataset_name}_{model_name}"
                model_temp_dir = output_dir / '_temp' / task_id
                (model_temp_dir / 'figures').mkdir(parents=True, exist_ok=True)
                (model_temp_dir / 'results').mkdir(parents=True, exist_ok=True)

                # Step 2: Dim Reduction & Clustering
                dim_results = run_dimensionality_reduction(
                    features, labels, class_names, model_temp_dir
                )

                if dim_results.get('reduced_features') is not None:
                     clustering_result = run_clustering_pipeline(
                         dim_results['reduced_features'],
                         labels,
                         class_names,
                         model_temp_dir,
                         fixed_k=None 
                    )


                # Step 3: Classification
                run_classification_pipeline(
                    features, labels, class_names, device, model_temp_dir, methods=['linear', 'knn']
                )

                # 归档
                final_res_dir = output_dir / 'results' / dataset_name / model_name
                final_fig_dir = output_dir / 'figures' / dataset_name / model_name
                final_res_dir.mkdir(parents=True, exist_ok=True)
                final_fig_dir.mkdir(parents=True, exist_ok=True)

                for f in (model_temp_dir / 'results').glob('*'):
                    shutil.copy(f, final_res_dir / f.name)
                for f in (model_temp_dir / 'figures').glob('*'):
                    shutil.copy(f, final_fig_dir / f.name)

        if (output_dir / '_temp').exists():
            shutil.rmtree(output_dir / '_temp')

        print(f"\n{'=' * 80}")
        print(f"所有任务完成！结果位于: {output_dir}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



