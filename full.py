import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Membaca data dari CSV
data = pd.read_csv('TpT.csv')

# Menampilkan data awal dan akhir untuk memastikan data terbaca dengan benar
print(data.head())
print(data.tail())

# Menyiapkan fitur untuk clustering
features = data[['Februari', 'Agustus']]

# Standarisasi fitur
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Menentukan jumlah cluster
num_clusters = 3

# Menerapkan K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Menampilkan hasil clustering
print(data)

# Visualisasi clustering
plt.figure(figsize=(10, 6))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('Clustering of Provinces by Unemployment Rates (February and August)')
plt.xlabel('Unemployment Rate - February (scaled)')
plt.ylabel('Unemployment Rate - August (scaled)')
plt.colorbar(label='Cluster')
plt.show()
