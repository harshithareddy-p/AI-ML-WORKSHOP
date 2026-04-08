import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'Area': [10, 20, 30, 70, 80, 90, 130, 150, 170],
    'HouseCost': [30, 40, 50, 70, 80, 90, 120, 130, 150]
})

inertia_values = []

for k in range(1, 8):
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(df)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 8), inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df)

plt.figure(figsize=(7, 5))
plt.scatter(df['Area'], df['HouseCost'], c=kmeans.labels_, s=80)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',
    s=200,
    color='red',
    label='Centroids'
)

plt.title("K-Means Clustering (K = 3)")
plt.xlabel("Area")
plt.ylabel("House Cost")
plt.legend()
plt.grid(True)
plt.show()

print("Inertia values for K = 1 to 7:", inertia_values)
print("Total points:", len(df))
print("Final inertia for K = 3:", kmeans.inertia_)
