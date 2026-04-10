import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
processed_data_path = "../../data/processed/universal_top_spotify_songs_cleaned.csv"
data = pd.read_csv(processed_data_path, skipinitialspace=True)
data = data.dropna()

# Reproduce K-Means
cluster_features = ['danceability', 'energy', 'tempo', 'loudness']
X_cluster = StandardScaler().fit_transform(data[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_cluster)

# Regression
X = data[['danceability', 'energy', 'tempo', 'loudness', 'is_explicit', 'release_year', 'cluster']]
y = data['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Linear Regression - MAE:", mean_absolute_error(y_test, lr_preds), "RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))
print("Random Forest - MAE:", mean_absolute_error(y_test, rf_preds), "RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))

# Classification
p50 = data['popularity'].quantile(0.50)
p90 = data['popularity'].quantile(0.90)

def classify_hit(score):
    if score < p50: return "Faible"
    elif score < p90: return "Moyen"
    else: return "Hit"

data['popularity_class'] = data['popularity'].apply(classify_hit)
y_class = data['popularity_class']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rfc.fit(Xc_train, yc_train)
rfc_preds = rfc.predict(Xc_test)
labels = ["Faible", "Moyen", "Hit"]
print("\nClassification Report:")
print(classification_report(yc_test, rfc_preds, labels=labels))