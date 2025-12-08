from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import save_figure


def handle_missing_data(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    missing_info = df.isnull().sum()
    print(f"Missing data:\n{missing_info}")
    
    if missing_info.sum() == 0:
        print("No missing data")
        return df
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, ax=ax, cmap='viridis')
    ax.set_title('Missing Data Heatmap')
    save_figure(fig, output_dir, '01_missing_data_heatmap')
    
    df_clean = df.fillna(method='ffill').fillna(method='bfill')
    
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    print(f"Remaining: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def detect_outliers(df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Detecting outliers (IQR)...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_mask = pd.DataFrame(index=df.index)
    outlier_stats = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask[col] = mask
        
        n_outliers = mask.sum()
        outlier_percentage = (n_outliers / len(df)) * 100
        
        outlier_stats.append({
            'column': col,
            'n_outliers': n_outliers,
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        print(f"{col}: {n_outliers} ({outlier_percentage:.2f}%)")
    
    outlier_stats_df = pd.DataFrame(outlier_stats)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:6]):
        if idx < len(axes):
            axes[idx].boxplot(df[col].dropna())
            axes[idx].set_title(f'{col}\nOutliers: {outlier_mask[col].sum()}')
            axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, output_dir, '02_outliers_boxplot')
    
    return outlier_mask, outlier_stats_df


def normalize_data(df: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    print(f"Normalizing ({method})...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print(f"Normalized {len(numeric_cols)} columns, shape: {df_normalized.shape}")
    
    return df_normalized, scaler


def analyze_correlations(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    print("Analyzing correlations...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    save_figure(fig, output_dir, '03_correlation_matrix')
    
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print(f"High correlations (|r| > 0.8): {len(high_corr_pairs)}")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print("No high correlations (|r| > 0.8)")
    
    return correlation_matrix


def apply_pca(df: pd.DataFrame, n_components: int = None, 
              variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, PCA, pd.DataFrame]:
    print("Applying PCA...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].values
    
    if n_components is None:
        pca_temp = PCA()
        pca_temp.fit(X)
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        print(f"Components: {n_components} (variance {variance_threshold:.1%})")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
    
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=pca_columns,
        index=numeric_cols
    )
    
    print(f"Features: {len(numeric_cols)} -> {n_components}")
    print(f"Variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca_df, pca, loadings_df


def create_feature_summary(df_original: pd.DataFrame, df_processed: pd.DataFrame,
                          outlier_stats: pd.DataFrame, correlation_matrix: pd.DataFrame) -> dict:
    print("Creating summary...")
    
    numeric_cols_original = df_original.select_dtypes(include=[np.number]).columns
    numeric_cols_processed = df_processed.select_dtypes(include=[np.number]).columns
    
    summary = {
        'original_shape': df_original.shape,
        'processed_shape': df_processed.shape,
        'numeric_features_original': len(numeric_cols_original),
        'numeric_features_processed': len(numeric_cols_processed),
        'missing_data_original': df_original.isnull().sum().sum(),
        'missing_data_processed': df_processed.isnull().sum().sum(),
        'total_outliers': outlier_stats['n_outliers'].sum(),
        'high_correlations': len(correlation_matrix[abs(correlation_matrix) > 0.8].sum()[correlation_matrix[abs(correlation_matrix) > 0.8].sum() > 1]),
        'feature_stats': {
            'original': {
                'mean': df_original[numeric_cols_original].mean().to_dict(),
                'std': df_original[numeric_cols_original].std().to_dict(),
                'min': df_original[numeric_cols_original].min().to_dict(),
                'max': df_original[numeric_cols_original].max().to_dict()
            },
            'processed': {
                'mean': df_processed[numeric_cols_processed].mean().to_dict(),
                'std': df_processed[numeric_cols_processed].std().to_dict(),
                'min': df_processed[numeric_cols_processed].min().to_dict(),
                'max': df_processed[numeric_cols_processed].max().to_dict()
            }
        }
    }
    
    print(f"Shape: {summary['original_shape']} -> {summary['processed_shape']}")
    print(f"Features: {summary['numeric_features_original']} -> {summary['numeric_features_processed']}")
    print(f"Outliers: {summary['total_outliers']}")
    
    return summary


def select_features(df: pd.DataFrame, correlation_matrix: pd.DataFrame, 
                   threshold: float = 0.95) -> pd.DataFrame:
    print(f"Selecting features (threshold {threshold})...")
    
    to_remove = set()
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                to_remove.add(correlation_matrix.columns[j])
    
    df_selected = df.drop(columns=list(to_remove))
    
    print(f"Removed {len(to_remove)} features: {df.shape[1]} -> {df_selected.shape[1]}")
    
    return df_selected


def normalize_features(df: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    return normalize_data(df, method)


def run_preprocessing_task(data_path: str, output_dir: Path) -> None:
    from utils import load_climate_data, save_dataframe
    
    print("[1/5] Loading")
    df = load_climate_data(data_path)
    print(f"Shape: {df.shape}")
    
    print("\n[2/5] Preprocessing")
    df_clean = handle_missing_data(df, output_dir)
    outlier_mask, outlier_stats = detect_outliers(df_clean, output_dir)
    
    non_outlier_mask = ~outlier_mask.any(axis=1)
    df_no_outliers = df_clean[non_outlier_mask]
    print(f"Removed {len(df_clean) - len(df_no_outliers)} outlier rows")
    
    print("\n[3/5] Correlation analysis")
    corr_matrix = analyze_correlations(df_no_outliers, output_dir)
    
    print("\n[4/5] Feature selection")
    df_selected = select_features(df_no_outliers, corr_matrix, threshold=0.95)
    
    print("\n[5/5] Normalization")
    df_normalized, scaler = normalize_features(df_selected, method='standard')
    
    save_dataframe(df_normalized, output_dir, 'data_preprocessed.csv')
    
    summary = create_feature_summary(df, df_normalized, outlier_stats, corr_matrix)
    summary['scaler'] = scaler
    summary['outlier_stats'] = outlier_stats.to_dict('records')
    summary['correlation_matrix'] = corr_matrix.to_dict()
    
    from utils import save_results
    save_results(summary, output_dir, 'preprocessing_summary.json')
    
    print(f"\nComplete. Output: {output_dir}")


def run_preprocessing_pipeline(df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, dict]:
    df_original = df.copy()
    
    df_clean = handle_missing_data(df, output_dir)
    outlier_mask, outlier_stats = detect_outliers(df_clean, output_dir)
    correlation_matrix = analyze_correlations(df_clean, output_dir)
    df_normalized, scaler = normalize_data(df_clean, method='standard')
    
    summary = create_feature_summary(df_original, df_normalized, outlier_stats, correlation_matrix)
    summary['scaler'] = scaler
    summary['outlier_stats'] = outlier_stats.to_dict('records')
    summary['correlation_matrix'] = correlation_matrix.to_dict()
    
    return df_normalized, summary

