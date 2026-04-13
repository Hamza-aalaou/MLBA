import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

processed_data_path = "/Users/hamza.aalaou/MLBA/dsas_template/data/processed/train_cleaned.csv"
data = pd.read_csv(processed_data_path, skipinitialspace=True)

numerical_cols = ['OverallQual', 'TotalSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']
model_data = data.dropna(subset=numerical_cols + ['SalePrice'])

X = model_data[numerical_cols]
y = model_data['SalePrice'] # Log1p transformed

lasso = LassoCV(cv=5, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def get_rmse_cv(model):
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
    return np.mean(-scores)

print(f"Lasso: {get_rmse_cv(lasso):.5f}")
print(f"RF: {get_rmse_cv(rf):.5f}")
print(f"XGB: {get_rmse_cv(xgb):.5f}")
