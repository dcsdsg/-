import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 默认黑体
plt.rcParams['axes.unicode_minus'] = False

# 自动确定项目路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
if os.path.basename(current_dir) == 'analysis':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

data_dir = os.path.join(project_root, 'data')
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

def load_data():
    train_path = os.path.join(data_dir, 'Dry_Bean_Dataset_Dirty_train.csv')
    val_path = os.path.join(data_dir, 'Dry_Bean_Dataset_Dirty_val.csv')
    test_path = os.path.join(data_dir, 'Dry_Bean_Dataset_Dirty_test.csv')
    
    # 路径防错回退机制
    if not os.path.exists(train_path):
        # 尝试在上一级目录或当前目录找
        alternative_data_dir = os.path.join(os.path.dirname(project_root), 'data')
        if os.path.exists(os.path.join(alternative_data_dir, 'Dry_Bean_Dataset_Dirty_train.csv')):
            train_path = os.path.join(alternative_data_dir, 'Dry_Bean_Dataset_Dirty_train.csv')
            val_path = os.path.join(alternative_data_dir, 'Dry_Bean_Dataset_Dirty_val.csv')
            test_path = os.path.join(alternative_data_dir, 'Dry_Bean_Dataset_Dirty_test.csv')
        else:
            # 尝试根目录
            train_path = 'Dry_Bean_Dataset_Dirty_train.csv'
            val_path = 'Dry_Bean_Dataset_Dirty_val.csv'
            test_path = 'Dry_Bean_Dataset_Dirty_test.csv'
            
    print(f"Loading data from:\n - {train_path}\n - {val_path}\n - {test_path}")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    return train, val, test

def missing_values_summary(df, name):
    print(f"--- Missing Values Summary for {name} ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if len(missing[missing > 0]) > 0 else "No missing values.")
    print("\n")

def identify_outliers_iqr(df, name):
    print(f"--- Outliers Summary for {name} ---")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            outlier_counts[col] = len(outliers)
    
    for col, count in outlier_counts.items():
        print(f"Column '{col}' has {count} outliers.")
    print("\n")

def plot_class_balance(train, val, test):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    if 'Class' in train.columns:
        sns.countplot(data=train, x='Class', order=sorted(train['Class'].dropna().unique()))
        plt.title('训练集类别分布 (Train Class Balance)')
        plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    if 'Class' in val.columns:
        sns.countplot(data=val, x='Class', order=sorted(val['Class'].dropna().unique()))
        plt.title('验证集类别分布 (Val Class Balance)')
        plt.xticks(rotation=45)
        
    plt.subplot(1, 3, 3)
    if 'Class' in test.columns:
        sns.countplot(data=test, x='Class', order=sorted(test['Class'].dropna().unique()))
        plt.title('测试集类别分布 (Test Class Balance)')
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'class_balance.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"类别分布图已保存至: {save_path}")

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    numerical_df = df.select_dtypes(include=[np.number]).replace('?', np.nan).apply(pd.to_numeric)
    numerical_df = numerical_df.fillna(numerical_df.median())
    
    # 算相关性
    corr = numerical_df.corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('干豆几何特征相关性热力图 (Correlation Heatmap)')
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'eda_correlation_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"相关性热力图已保存至: {save_path}")

def plot_feature_kde_overlay(train, val, test, features=['Area', 'Perimeter', 'roundness', 'Solidity']):
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 2, i+1)
        
        # 预处理为数值型并填缺失值
        t_col = pd.to_numeric(train[col], errors='coerce')
        t_col = t_col.fillna(t_col.median())
        v_col = pd.to_numeric(val[col], errors='coerce')
        v_col = v_col.fillna(v_col.median())
        te_col = pd.to_numeric(test[col], errors='coerce')
        te_col = te_col.fillna(te_col.median())
        
        sns.kdeplot(t_col, label='Train', fill=True, alpha=0.3, color='skyblue')
        sns.kdeplot(v_col, label='Val', fill=True, alpha=0.3, color='orange')
        sns.kdeplot(te_col, label='Test', fill=True, alpha=0.3, color='green')
        
        plt.title(f'{col} 在不同数据集上的 KDE 分布')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'eda_feature_kde.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"特征数据集KDE对比图已保存至: {save_path}")

def plot_feature_boxplot_by_class(df, features=['Area', 'roundness', 'Solidity', 'AspectRation']):
    plt.figure(figsize=(12, 10))
    # 预处理
    df_clean = df.copy()
    for col in features:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    classes_order = sorted(df_clean['Class'].dropna().unique())
    
    for i, col in enumerate(features):
        if col in df_clean.columns:
            plt.subplot(2, 2, i+1)
            sns.boxplot(data=df_clean, x='Class', y=col, order=classes_order)
            plt.title(f'特征 {col} 按类别箱线图 (Boxplot by Class)')
            plt.xticks(rotation=45)
            
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'eda_feature_boxplot.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"按类别箱线图已保存至: {save_path}")

def plot_pca_tsne_visualization(df):
    # 数据清洗与标准化
    X = df.drop(columns=['Class']).replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    y = df['Class'].fillna('Unknown')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA 降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 随机取样 1000 个样本绘制 t-SNE (避免运行缓慢)
    sample_size = min(1000, len(df))
    indices = np.random.choice(len(df), sample_size, replace=False)
    X_scaled_sample = X_scaled[indices]
    y_sample = y.iloc[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled_sample)
    
    # 开始画图
    plt.figure(figsize=(16, 7))
    
    # PCA 散点图
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set2', alpha=0.7)
    plt.title('干豆数据集特征空间的 PCA 二维投影')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # t-SNE 散点图
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_sample, palette='Set2', alpha=0.7)
    plt.title('干豆数据集特征空间的 t-SNE 二维投影 (采样1000个样本)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'eda_pca_tsne_2d.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PCA/t-SNE 二维投影图已保存至: {save_path}")

def plot_pca_variance_ratio(df):
    X = df.drop(columns=['Class']).replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    exp_var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var_ratio)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(exp_var_ratio) + 1), exp_var_ratio, alpha=0.5, align='center',
            label='Individual Explained Variance', color='skyblue')
    plt.step(range(1, len(cum_var_ratio) + 1), cum_var_ratio, where='mid',
             label='Cumulative Explained Variance', color='red', linewidth=2)
    
    # 画一条 95% 方差解释率的虚线
    plt.axhline(y=0.95, color='gray', linestyle='--', label='95% Threshold')
    
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title('PCA 方差贡献率与累积方差贡献率 (PCA Variance Plot)')
    plt.xticks(range(1, len(exp_var_ratio) + 1))
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'eda_pca_variance.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PCA 方差贡献率图已保存至: {save_path}")

def clean_dataset_features_and_target(df):
    df = df.copy()
    
    # 1. Clean target Class
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(str).str.strip().str.upper()
        typo_map = {
            'D3RMAS0N': 'DERMASON',
            'S3K3R': 'SEKER',
            'B0MBAY': 'BOMBAY',
            'H0R0Z': 'HOROZ'
        }
        df['Class'] = df['Class'].replace(typo_map)
        
    # 2. Clean feature columns (e.g., Solidity "?" or Compactness "cm")
    for col in df.columns:
        if col != 'Class':
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
                # Remove any non-numeric suffixes
                df[col] = df[col].str.replace(r'[^\d.\-]', '', regex=True)
                df[col] = df[col].replace(['', 'nan', 'None'], np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Recover negative Area values
    if 'Area' in df.columns:
        df['Area'] = df['Area'].abs()
        
    # 4. Mathematically recover Solidity and Perimeter
    if 'Solidity' in df.columns and 'Area' in df.columns and 'ConvexArea' in df.columns:
        df['Solidity'] = df['Solidity'].fillna(df['Area'] / df['ConvexArea'])
        
    if 'Perimeter' in df.columns and 'Area' in df.columns and 'roundness' in df.columns:
        df['Perimeter'] = df['Perimeter'].fillna(np.sqrt(4 * np.pi * df['Area'] / df['roundness']))
        
    # 5. Generic fallback imputation
    df = df.fillna(df.median(numeric_only=True))
    
    return df

if __name__ == "__main__":
    print("=== Start EDA Analysis ===")
    train, val, test = load_data()
    
    # 打印数据分类污染情况（拼写错误）作为论文排查素材
    if 'Class' in train.columns:
        print("--- Target Class Spelling Pollution (Before Cleaning) ---")
        print(f"Number of unique classes in Train: {train['Class'].nunique()}")
        print(f"Unique classes: {train['Class'].unique().tolist()}")
        print("\n")
        
    # 打印数据特征污染情况（单位后缀）作为论文排查素材
    if 'Compactness' in train.columns:
        print("--- Feature Suffix Unit Pollution (Before Cleaning) ---")
        cm_count = train['Compactness'].astype(str).str.contains('cm').sum()
        print(f"Number of rows in Compactness with 'cm' unit suffix: {cm_count}")
        print(f"Example polluted values: {train[train['Compactness'].astype(str).str.contains('cm')]['Compactness'].unique()[:5].tolist()}")
        print("\n")

    # 缺失值与离群点分析
    missing_values_summary(train, 'Train')
    missing_values_summary(val, 'Validation')
    missing_values_summary(test, 'Test')
    
    identify_outliers_iqr(train, 'Train')
    
    # 清洗数据特征和类别以便绘制整洁的图表
    train = clean_dataset_features_and_target(train)
    val = clean_dataset_features_and_target(val)
    test = clean_dataset_features_and_target(test)
    
    # 可视化图表生成
    print("Generating EDA Plots...")
    plot_class_balance(train, val, test)
    plot_correlation_heatmap(train)
    plot_feature_kde_overlay(train, val, test)
    plot_feature_boxplot_by_class(train)
    plot_pca_tsne_visualization(train)
    plot_pca_variance_ratio(train)
    
    print("=== EDA Analysis Complete successfully! ===")
