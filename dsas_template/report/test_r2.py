import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../data/processed/universal_top_spotify_songs_cleaned.csv", skipinitialspace=True)
df = df.dropna()

cluster_features = ['danceability', 'energy', 'tempo', 'loudness']
X_cluster = StandardScaler().fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster)

X = df[['danceability', 'energy', 'tempo', 'loudness', 'is_explicit', 'release_year', 'cluster']]
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print("LR R2:", r2_score(y_test, lr_preds))

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("RF R2:", r2_score(y_test, rf_preds))
