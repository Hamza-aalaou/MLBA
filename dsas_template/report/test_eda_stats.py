import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("../../data/processed/universal_top_spotify_songs_cleaned.csv", skipinitialspace=True)
df = df.dropna()

print("=== Q1: POPULARITY DISTRIBUTION ===")
print(df['popularity'].describe())
print("Skewness:", df['popularity'].skew())

print("\n=== Q2: CORRELATIONS ===")
cols = ['popularity', 'popularity_zscore', 'danceability', 'energy', 'tempo', 'loudness', 'is_explicit']
print(df[cols].corr()['popularity'])

print("\n=== Q3: K-MEANS CLUSTERS ===")
cluster_features = ['danceability', 'energy', 'tempo', 'loudness']
X_cluster = StandardScaler().fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster)
print(df.groupby('cluster')[cluster_features + ['popularity']].mean())
