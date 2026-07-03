import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

class DryBeanDataset(Dataset):
    def __init__(self, data, target_col='Class', is_train=True, n_components=None, 
                 scaler=None, pca=None, label_encoder=None, pre_cleaned=False):
        """
        data: pandas DataFrame or string path to CSV
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
            
        self.target_col = target_col
        self.is_train = is_train
        
        # Split features and target
        if target_col in df.columns:
            y_raw = df[target_col]
            if pre_cleaned:
                X_raw = df.drop(columns=[target_col])
            else:
                X_raw = df.drop(columns=[target_col]).replace('?', np.nan)
                
                # Clean target class labels (lowercase, whitespaces, leetspeak spelling errors)
                y_raw = y_raw.astype(str).str.strip().str.upper()
                typo_map = {
                    'D3RMAS0N': 'DERMASON',
                    'S3K3R': 'SEKER',
                    'B0MBAY': 'BOMBAY',
                    'H0R0Z': 'HOROZ'
                }
                y_raw = y_raw.replace(typo_map)
        else:
            y_raw = None
            X_raw = df
            if not pre_cleaned:
                X_raw = X_raw.replace('?', np.nan)
            
        if not pre_cleaned:
            # Clean numeric columns containing text suffixes or unit strings (e.g., "0.9293 cm", "?")
            for col in X_raw.columns:
                if X_raw[col].dtype == 'object' or pd.api.types.is_string_dtype(X_raw[col]):
                    X_raw[col] = X_raw[col].astype(str).str.strip()
                    # Remove any characters other than digits, decimal point, and minus sign
                    X_raw[col] = X_raw[col].str.replace(r'[^\d.\-]', '', regex=True)
                    X_raw[col] = X_raw[col].replace(['', 'nan', 'None'], np.nan)

            # Convert to numeric
            X_raw = X_raw.apply(pd.to_numeric, errors='coerce')

            # 1. Mathematical Recovery of features before standard imputation
            # Correct negative Area values (simple typographical minus sign)
            if 'Area' in X_raw.columns:
                X_raw['Area'] = X_raw['Area'].abs()
                
            # Mathematically reconstruct missing Solidity (Solidity = Area / ConvexArea)
            if 'Solidity' in X_raw.columns and 'Area' in X_raw.columns and 'ConvexArea' in X_raw.columns:
                X_raw['Solidity'] = X_raw['Solidity'].fillna(X_raw['Area'] / X_raw['ConvexArea'])
                
            # Mathematically reconstruct missing Perimeter (Perimeter = sqrt(4 * pi * Area / roundness))
            if 'Perimeter' in X_raw.columns and 'Area' in X_raw.columns and 'roundness' in X_raw.columns:
                X_raw['Perimeter'] = X_raw['Perimeter'].fillna(np.sqrt(4 * np.pi * X_raw['Area'] / X_raw['roundness']))

            # 2. Fallback Imputation (fillna with median for any remaining NaNs)
            X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
        
        # 3. Handle outliers (cap at 3-sigma)
        for col in X_raw.columns:
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                mean = X_raw[col].mean()
                std = X_raw[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                X_raw[col] = X_raw[col].clip(lower=lower_bound, upper=upper_bound)
                
        # 3. & 4. StandardScaler and PCA
        if self.is_train:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_raw)
            
            if n_components is not None:
                self.pca = PCA(n_components=n_components)
                X_processed = self.pca.fit_transform(X_scaled)
            else:
                self.pca = None
                X_processed = X_scaled
                
            if y_raw is not None:
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y_raw)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for evaluation.")
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_raw)
            
            if pca is not None:
                self.pca = pca
                X_processed = self.pca.transform(X_scaled)
            else:
                self.pca = None
                X_processed = X_scaled
                
            if y_raw is not None:
                if label_encoder is None:
                    raise ValueError("Label encoder must be provided for evaluation.")
                self.label_encoder = label_encoder
                y_processed = self.label_encoder.transform(y_raw)
                
        self.X = torch.tensor(X_processed, dtype=torch.float32)
        if y_raw is not None:
            self.y = torch.tensor(y_processed, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def get_dataloader(data, batch_size=32, shuffle=True, **kwargs):
    dataset = DryBeanDataset(data, **kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

if __name__ == '__main__':
    # Create dummy data to test
    np.random.seed(42)
    num_samples = 100
    features = pd.DataFrame(np.random.randn(num_samples, 10), columns=[f'feature_{i}' for i in range(10)])
    # Add some NaNs and outliers
    features.iloc[0, 0] = np.nan
    features.iloc[1, 1] = 1000.0  # outlier
    
    target = np.random.choice(['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI'], size=num_samples)
    
    df = features.copy()
    df['Class'] = target
    
    print("Testing DataLoader with dummy data...")
    print(f"Original shape: {df.shape}")
    
    dataloader, dataset = get_dataloader(df, batch_size=16, is_train=True, n_components=5)
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"X_batch shape: {X_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        break  # Just check the first batch
    
    print("Test completed successfully.")
