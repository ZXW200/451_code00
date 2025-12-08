import argparse
import sys
import traceback
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# 引入 kagglehub 用于自动下载 Food-101
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

# 1. 选择要运行的数据集
TARGET_DATASETS = ['oxford_pets', 'food101']

# 2. 选择要使用的模型 (这里修改为包含所有三个模型)
MODELS_TO_RUN = ['resnet50', 'densenet121', 'dinov2']

# 3. 数据集存放根目录 (仅用于 Oxford Pets，Food-101 会被 kagglehub 存放在缓存目录)
DATA_ROOT = Path("datasets")


# ==========================================

class DownloadProgressBar(tqdm):
    """用于 urllib 的下载进度条"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def reorganize_oxford_pets(images_dir: Path):
    """整理 Oxford Pets 文件结构"""
    print("正在整理 Oxford Pets 文件结构...")
    images = list(images_dir.glob('*.jpg'))
    if not images:
        return

    count = 0
    for img_path in images:
        filename = img_path.name
        # 文件名格式通常为: Class_Name_Number.jpg
        # 我们提取最后一个下划线前的部分作为类别名
        parts = filename.rsplit('_', 1)
        if len(parts) != 2:
            continue

        class_name = parts[0].title()
        target_dir = images_dir / class_name
        target_dir.mkdir(exist_ok=True)

        try:
            shutil.move(str(img_path), str(target_dir / filename))
            count += 1
        except Exception:
            pass
    print(f"整理完成：已归类 {count} 张图片。")


def prepare_oxford_pets(root_dir: Path) -> str:
    """处理 Oxford Pets (直链下载)"""
    base_dir = root_dir / 'oxford_pets'
    base_dir.mkdir(parents=True, exist_ok=True)

    target_img_dir = base_dir / 'images'

    # 检查是否已准备好
    if target_img_dir.exists() and any(target_img_dir.iterdir()):
        first_item = next(target_img_dir.iterdir())
        if first_item.is_dir():
            print(f"本地数据集 [Oxford Pets] 已就绪: {target_img_dir}")
            return str(target_img_dir)
        else:
            reorganize_oxford_pets(target_img_dir)
            return str(target_img_dir)

    # 下载
    url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
    tar_path = base_dir / "images.tar.gz"

    print(f"正在下载 Oxford Pets...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Oxford Pets') as t:
            urllib.request.urlretrieve(url, filename=tar_path, reporthook=t.update_to)

        print("正在解压...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=base_dir)

        os.remove(tar_path)  # 删除压缩包

        # 整理
        reorganize_oxford_pets(target_img_dir)
        return str(target_img_dir)

    except Exception as e:
        print(f"Oxford Pets 下载失败: {e}")
        raise e


def prepare_food101() -> str:
    """使用 kagglehub 自动下载 Food-101"""
    print("\n正在通过 kagglehub 获取 Food-101 数据集...")
    print("如果是首次下载，可能需要一点时间...")

    try:
        # kagglehub 会自动下载最新版并返回路径
        path = kagglehub.dataset_download("dansbecker/food-101")
        print(f"Food-101 数据集路径: {path}")

        path_obj = Path(path)

        # 寻找包含图片的 images 文件夹
        if (path_obj / 'images').exists():
            return str(path_obj / 'images')
        elif (path_obj / 'food-101' / 'images').exists():
            return str(path_obj / 'food-101' / 'images')

        # 兜底：递归查找 images 文件夹
        for root, dirs, files in os.walk(path):
            if 'images' in dirs:
                return os.path.join(root, 'images')

        return str(path)

    except Exception as e:
        print(f"kagglehub 下载失败: {e}")
        print("请检查网络或配置 Kaggle API Token")
        raise e


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    # 依然允许命令行覆盖模型列表
    parser.add_argument('--models', type=str, nargs='+', default=MODELS_TO_RUN, help='Override models list')
    args = parser.parse_args()

    models_to_run = args.models if args.models else MODELS_TO_RUN

    try:
        output_dir = create_output_dir('pipeline', 'history')
        device = get_device()
        batch_size = calculate_batch_size(device)

        print(f"输出目录: {output_dir}")
        print(f"计算设备: {device}")
        print(f"将要运行的模型: {models_to_run}")

        for subdir in ['figures', 'results']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # === 遍历数据集 ===
        for dataset_name in TARGET_DATASETS:
            print(f"\n{'=' * 60}")
            print(f"准备处理数据集: {dataset_name}")
            print(f"{'=' * 60}")

            # 1. 根据数据集名称选择不同的准备策略
            if dataset_name == 'oxford_pets':
                dataset_path_str = prepare_oxford_pets(DATA_ROOT)
            elif dataset_name == 'food101':
                dataset_path_str = prepare_food101()
            else:
                print(f"跳过未知数据集: {dataset_name}")
                continue

            # === 遍历模型 ===
            for model_idx, model_name in enumerate(models_to_run, 1):
                print(f"\n--- 模型 [{model_name}] ({model_idx}/{len(models_to_run)}) 处理 {dataset_name} ---")

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

                # 临时目录
                task_id = f"{dataset_name}_{model_name}"
                model_temp_dir = output_dir / '_temp' / task_id
                (model_temp_dir / 'figures').mkdir(parents=True, exist_ok=True)
                (model_temp_dir / 'results').mkdir(parents=True, exist_ok=True)

                # Step 2: Dim Reduction & Clustering
                dim_results = run_dimensionality_reduction(
                    features, labels, class_names, model_temp_dir
                )

                if dim_results.get('reduced_features') is not None:
                    run_clustering_pipeline(
                        dim_results['reduced_features'], labels, class_names, model_temp_dir
                    )

                # Step 3: Classification
                run_classification_pipeline(
                    features, labels, class_names, device, model_temp_dir, methods=['linear', 'knn']
                )

                # 归档结果
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
