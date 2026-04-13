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

# Fig 1
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(raw_data['SalePrice'], bins=30, kde=True, ax=ax[0], color='skyblue')
ax[0].set_title("Raw SalePrice Distribution")
ax[0].set_xlabel("Sale Price ($)")
sns.histplot(data['SalePrice'], bins=30, kde=True, ax=ax[1], color='coral')
ax[1].set_title("Log1p Transformed SalePrice")
ax[1].set_xlabel("Log(1 + SalePrice)")
plt.tight_layout()
plt.savefig('/Users/hamza.aalaou/MLBA/Template_for_Academic_Journal_on_Computing__Engineering_and_Applied_Mathematics__AJCEAM___1_/fig-1.png', dpi=300)
plt.close()

# Fig 2
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
plt.savefig('/Users/hamza.aalaou/MLBA/Template_for_Academic_Journal_on_Computing__Engineering_and_Applied_Mathematics__AJCEAM___1_/fig-2.png', dpi=300)
plt.close()
