import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind", font_scale=1.1)

processed_data_path = "/Users/hamza.aalaou/MLBA/dsas_template/data/processed/train_cleaned.csv"
raw_data_path = "/Users/hamza.aalaou/MLBA/dsas_template/data/raw/train.csv"
data = pd.read_csv(processed_data_path, skipinitialspace=True)
raw_data = pd.read_csv(raw_data_path, skipinitialspace=True)

# 1. fig-saleprice-dist
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(raw_data['SalePrice'], bins=30, kde=True, ax=ax[0], color='skyblue')
ax[0].set_title("Raw SalePrice Distribution")
ax[0].set_xlabel("Sale Price ($)")
sns.histplot(data['SalePrice'], bins=30, kde=True, ax=ax[1], color='coral')
ax[1].set_title("Log1p Transformed SalePrice")
ax[1].set_xlabel("Log(1 + SalePrice)")
plt.tight_layout()
plt.savefig('/Users/hamza.aalaou/MLBA/fig-saleprice-dist.png', dpi=300)
plt.close()

# 2. fig-correlation-matrix
features = ['SalePrice', 'OverallQual', 'TotalSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']
corr_matrix = data[features].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Key Property Features")
plt.tight_layout()
plt.savefig('/Users/hamza.aalaou/MLBA/fig-correlation-matrix.png', dpi=300)
plt.close()

# 3. fig-outlier-detection
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=raw_data, color='purple', alpha=0.6)
plt.axvline(x=4000, color='red', linestyle='--', label='GrLivArea > 4000')
plt.title("Identifying 'Luxury Outliers'")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Raw Sale Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig('/Users/hamza.aalaou/MLBA/fig-outlier-detection.png', dpi=300)
plt.close()

# 4. fig-neighborhood-price
plt.figure(figsize=(14, 6))
sorted_nb = raw_data.groupby('Neighborhood')['SalePrice'].median().sort_values().index
sns.boxplot(x='Neighborhood', y='SalePrice', data=raw_data, order=sorted_nb, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title("Impact of Neighborhood on Sale Price")
plt.ylabel("Raw Sale Price ($)")
plt.xlabel("Neighborhood")
plt.tight_layout()
plt.savefig('/Users/hamza.aalaou/MLBA/fig-neighborhood-price.png', dpi=300)
plt.close()

# 5. fig-year-built
plt.figure(figsize=(10, 5))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=raw_data, alpha=0.4, color='teal')
sns.lineplot(x='YearBuilt', y=raw_data.groupby('YearBuilt')['SalePrice'].transform('median'), data=raw_data, color='red', label='Median Price Trend')
plt.title("Sale Price vs. Original Construction Year")
plt.xlabel("Year Built")
plt.ylabel("Raw Sale Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig('/Users/hamza.aalaou/MLBA/fig-year-built.png', dpi=300)
plt.close()

# 6. unsupervised-learning
qual_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd'] 
pca_data = data[['SalePrice'] + qual_features].dropna()
pca = PCA(n_components=1)
pca_data['AgeQualityIndex'] = pca.fit_transform(StandardScaler().fit_transform(pca_data[qual_features]))
kmeans = KMeans(n_clusters=3, random_state=42)
pca_data['MarketSegment'] = kmeans.fit_predict(pca_data[['AgeQualityIndex', 'SalePrice']])
plt.figure(figsize=(8, 6))
sns.scatterplot(x='AgeQualityIndex', y='SalePrice', hue='MarketSegment', palette='Set1', data=pca_data)
plt.title("Market Segments via K-Means and PCA")
plt.xlabel("Age-Quality Index (PCA Component 1)")
plt.ylabel("Log1p(SalePrice)")
plt.tight_layout()
plt.savefig('/Users/hamza.aalaou/MLBA/unsupervised-learning.png', dpi=300)
plt.close()
print("All figures successfully exported to /Users/hamza.aalaou/MLBA/")
