import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Establish the project root directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Paths
    raw_train_path = PROJECT_ROOT / 'dsas_template' / 'data' / 'raw' / 'train.csv'
    raw_test_path = PROJECT_ROOT / 'dsas_template' / 'data' / 'raw' / 'test.csv'
    processed_train_path = PROJECT_ROOT / 'dsas_template' / 'data' / 'processed' / 'train_cleaned.csv'
    processed_test_path = PROJECT_ROOT / 'dsas_template' / 'data' / 'processed' / 'test_cleaned.csv'
    
    print(f"Loading data from: {raw_train_path}")
    
    # Load raw data
    try:
        train_df = pd.read_csv(raw_train_path)
        test_df = pd.read_csv(raw_test_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Original train dataset shape: {train_df.shape}")
    print(f"Original test dataset shape: {test_df.shape}")

    # Helper function for data cleaning following scientific practices
    def clean_dataset(df, is_train=True):
        df_clean = df.copy()

        # 1. Handling Missingness scientifically
        # Many "missing" values in categorical variables denote the absence of the feature.
        # Imputing these 'NA'/'NaN' with the string 'None' (e.g. PoolQC, MiscFeature, Alley, etc.)
        cols_with_none_meaning = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'
        ]
        for col in cols_with_none_meaning:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('None')

        # 2. Skewness in Target Variable
        # For the target variable (SalePrice), it typically exhibits right-skewness.
        # We apply Log Transformation to normalize its distribution (standard practice in econometrics).
        if is_train and 'SalePrice' in df_clean.columns:
            df_clean['SalePrice'] = np.log1p(df_clean['SalePrice'])

        # 3. Feature Engineering: New Features
        # Create a TotalSF (Total Square Footage) feature by summing basement and ground-level areas.
        # Note: BsmtFinSF1, BsmtFinSF2, BsmtUnfSF sum to TotalBsmtSF, 
        # and 1stFlrSF, 2ndFlrSF, LowQualFinSF sum to GrLivArea.
        # The prompt specifically requests summing basement and ground-level areas.
        
        # We will assure numeric filling for area sums if they miss any value
        total_bsmt_sf = df_clean['TotalBsmtSF'].fillna(0)
        gr_liv_area = df_clean['GrLivArea'].fillna(0)
        
        df_clean['TotalSF'] = total_bsmt_sf + gr_liv_area

        return df_clean

    train_cleaned = clean_dataset(train_df, is_train=True)
    test_cleaned = clean_dataset(test_df, is_train=False)

    print(f"Cleaned train dataset shape: {train_cleaned.shape}")
    print(f"Cleaned test dataset shape: {test_cleaned.shape}")

    # Save cleaned data
    processed_train_path.parent.mkdir(parents=True, exist_ok=True)
    processed_test_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_cleaned.to_csv(processed_train_path, index=False)
    test_cleaned.to_csv(processed_test_path, index=False)
    
    print(f"Cleaned train data saved to: {processed_train_path}")
    print(f"Cleaned test data saved to: {processed_test_path}")

if __name__ == '__main__':
    main()
