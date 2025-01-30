import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Fetch the data from the URL
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Mall_Customers-UZjOjWThOtXGch6V820TFrmRsIvZg2.csv"
response = requests.get(url)
data = StringIO(response.text)

# Load the data into a pandas DataFrame
df = pd.read_csv(data)

# Display basic information about the dataset
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Preprocess the data
df['Age'] = pd.to_numeric(df['Age'])
df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'])
df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'])

# Select features for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_curve.png')
plt.close()

# Perform K-means clustering with the optimal number of clusters (let's say 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.colorbar(scatter)
plt.savefig('customer_segments.png')
plt.close()

# Analyze the clusters
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})

print("\nCluster Summary:")
print(cluster_summary)

# Print the results
print("\nClustering complete. Results saved in 'elbow_curve.png' and 'customer_segments.png'.")
print("Cluster summary provides an overview of each customer segment.")