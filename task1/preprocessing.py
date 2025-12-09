from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import save_fig


# Fill missing values in dataframe using forward and backward fill
def fill_na(df, out_path):
    nulls = df.isnull().sum()
    print(f"NaNs:\n{nulls}")

    # Check if there are any missing values
    if nulls.sum() == 0:
        print("No NaNs")
        return df

    # Create heatmap visualization of missing data
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, ax=ax, cmap='viridis')
    ax.set_title('Missing Map')
    save_fig(fig, out_path, '01_missing')

    # Forward fill then backward fill missing values
    df_new = df.fillna(method='ffill').fillna(method='bfill')

    # Fill any remaining NaN with column mean for numeric columns
    for c in df_new.select_dtypes(include=[np.number]).columns:
        if df_new[c].isnull().sum() > 0:
            df_new[c].fillna(df_new[c].mean(), inplace=True)

    print(f"Left NaNs: {df_new.isnull().sum().sum()}")

    return df_new


# Detect outliers using IQR method
def find_outliers(df, out_path):
    print("Checking outliers...")

    num_cols = df.select_dtypes(include=[np.number]).columns
    mask = pd.DataFrame(index=df.index)
    stats = []

    # Calculate IQR-based outlier boundaries for each numeric column
    for c in num_cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        # Identify outliers outside the boundaries
        is_out = (df[c] < low) | (df[c] > high)
        mask[c] = is_out

        cnt = is_out.sum()
        pct = (cnt / len(df)) * 100

        # Store outlier statistics
        stats.append({
            'col': c,
            'count': cnt,
            'pct': pct,
            'low': low,
            'high': high
        })

        print(f"{c}: {cnt} ({pct:.2f}%)")

    stats_df = pd.DataFrame(stats)

    # Create boxplot visualizations for first 6 numeric columns
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    ax = ax.flatten()

    for i, c in enumerate(num_cols[:6]):
        if i < len(ax):
            ax[i].boxplot(df[c].dropna())
            ax[i].set_title(f'{c}\nOut: {mask[c].sum()}')
            ax[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(num_cols), len(ax)):
        ax[i].set_visible(False)

    plt.tight_layout()
    save_fig(fig, out_path, '02_outliers')

    return mask, stats_df


# Scale numeric data using standard or minmax normalization
def scale_data(df, mode):
    print(f"Scaling ({mode})...")

    cols = df.select_dtypes(include=[np.number]).columns

    # Choose scaler based on mode
    if mode == 'standard':
        scaler = StandardScaler()
    elif mode == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Mode error")

    # Apply scaling to numeric columns
    df_sc = df.copy()
    df_sc[cols] = scaler.fit_transform(df[cols])

    print(f"Scaled {len(cols)} cols, shape: {df_sc.shape}")

    return df_sc, scaler


# Calculate and visualize correlation matrix
def calc_corr(df, out_path):
    print("Calc correlations...")

    cols = df.select_dtypes(include=[np.number]).columns
    mat = df[cols].corr()

    # Create heatmap of correlation matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(mat, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax, fmt='.2f')
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    save_fig(fig, out_path, '03_corr')

    # Find highly correlated feature pairs
    high_c = []
    for i in range(len(mat.columns)):
        for j in range(i+1, len(mat.columns)):
            val = mat.iloc[i, j]
            if abs(val) > 0.8:
                high_c.append({
                    'f1': mat.columns[i],
                    'f2': mat.columns[j],
                    'val': val
                })

    # Print high correlation pairs
    if high_c:
        print(f"High corr: {len(high_c)}")
        for p in high_c:
            print(f"  {p['f1']} - {p['f2']}: {p['val']:.3f}")
    else:
        print("No high corr")

    return mat


# Perform PCA dimensionality reduction
def run_pca(df, n_comp, var_th) :
    print("Running PCA...")

    cols = df.select_dtypes(include=[np.number]).columns
    X = df[cols].values

    # Automatically determine number of components if not specified
    if n_comp is None:
        tmp = PCA()
        tmp.fit(X)
        cum = np.cumsum(tmp.explained_variance_ratio_)
        n_comp = np.argmax(cum >= var_th) + 1
        print(f"Selected n={n_comp} (var {var_th:.1%})")

    # Apply PCA transformation
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    # Create dataframe with principal components
    names = [f'PC{i+1}' for i in range(n_comp)]
    res = pd.DataFrame(X_pca, columns=names, index=df.index)

    # Get feature loadings
    loads = pd.DataFrame(
        pca.components_.T,
        columns=names,
        index=cols
    )

    print(f"Feats: {len(cols)} -> {n_comp}")
    print(f"Var: {pca.explained_variance_ratio_.sum():.3f}")

    return res, pca, loads


# Generate summary statistics comparing raw and processed data
def get_summary(df1, df2, outliers, corr) :
    print("Summarizing...")

    cols1 = df1.select_dtypes(include=[np.number]).columns
    cols2 = df2.select_dtypes(include=[np.number]).columns

    # Compile preprocessing statistics
    info = {
        'shape_raw': df1.shape,
        'shape_proc': df2.shape,
        'n_feats_raw': len(cols1),
        'n_feats_proc': len(cols2),
        'nan_raw': df1.isnull().sum().sum(),
        'nan_proc': df2.isnull().sum().sum(),
        'n_out': outliers['count'].sum(),
        'n_high_corr': len(corr[abs(corr) > 0.8].sum()[corr[abs(corr) > 0.8].sum() > 1]),
        'stats': {
            'raw': {
                'mean': df1[cols1].mean().to_dict(),
                'std': df1[cols1].std().to_dict(),
                'min': df1[cols1].min().to_dict(),
                'max': df1[cols1].max().to_dict()
            },
            'proc': {
                'mean': df2[cols2].mean().to_dict(),
                'std': df2[cols2].std().to_dict(),
                'min': df2[cols2].min().to_dict(),
                'max': df2[cols2].max().to_dict()
            }
        }
    }

    print(f"Shape: {info['shape_raw']} -> {info['shape_proc']}")
    print(f"Feats: {info['n_feats_raw']} -> {info['n_feats_proc']}")
    print(f"Outliers: {info['n_out']}")

    return info


# Remove highly correlated features based on threshold
def filter_feats(df, corr, th) :
    print(f"Selecting feats (th={th})...")

    drop_list = set()

    # Identify features to drop from correlation pairs
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > th:
                drop_list.add(corr.columns[j])

    # Remove selected features
    res = df.drop(columns=list(drop_list))

    print(f"Dropped {len(drop_list)} cols: {df.shape[1]} -> {res.shape[1]}")

    return res


# Main preprocessing pipeline function
def run_prep(path, out_dir) :
    from utils import load_data, save_csv

    # Load raw data
    print("Loading")
    df = load_data(path)
    print(f"Shape: {df.shape}")

    # Clean data by filling missing values and removing outliers
    print("\nCleaning")
    df_cl = fill_na(df, out_dir)
    mask, stats = find_outliers(df_cl, out_dir)

    # Remove rows with outliers
    ok = ~mask.any(axis=1)
    df_ok = df_cl[ok]
    print(f"Removed {len(df_cl) - len(df_ok)} rows")

    # Calculate feature correlations
    print("\nCorrelation")
    corr = calc_corr(df_ok, out_dir)

    # Select features by removing highly correlated ones
    print("\nSelection")
    df_sel = filter_feats(df_ok, corr, th=0.95)

    # Normalize data using standard scaling
    print("\nNormalization")
    df_fin, sc = scale_data(df_sel, mode='standard')

    # Save cleaned data to CSV
    save_csv(df_fin, out_dir, 'clean_data.csv')

    # Generate and save preprocessing summary
    summ = get_summary(df, df_fin, stats, corr)
    summ['scaler'] = sc
    summ['out_stats'] = stats.to_dict('records')
    summ['corr'] = corr.to_dict()

    from utils import save_json
    save_json(summ, out_dir, 'summary.json')

    print(f"\nDone. Output: {out_dir}")

