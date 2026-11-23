# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the customer dataset containing features such as Annual Income and Spending Score.
2. Select the required features and train the K-Means Clustering model by specifying the number of clusters.
3. Assign each customer to a cluster and determine the cluster centroids.
4. Display the customer clusters and visualize the clustered data with the centroids.

## Program:

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Sample Dataset
# Features: Annual Income (k$), Spending Score
X = np.array([
    [15, 39],
    [16, 81],
    [17, 6],
    [18, 77],
    [20, 40],
    [25, 76],
    [30, 6],
    [35, 72],
    [40, 55],
    [45, 30],
    [50, 50],
    [55, 60],
    [60, 40],
    [65, 70],
    [70, 20],
    [75, 80],
    [80, 35],
    [85, 90],
    [90, 15],
    [95, 95]
])

# K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

kmeans.fit(X)

# Cluster Labels
labels = kmeans.labels_

centroids = kmeans.cluster_centers_

print("Cluster Labels:")

print(labels)

print("\nCluster Centroids:")

print(np.round(centroids, 2))

# Plot Clusters
plt.figure(figsize=(8,6))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=80)

plt.scatter(centroids[:, 0], centroids[:, 1],
            color='red', marker='X', s=250, label='Centroids')

plt.title("K-Means Clustering for Customer Segmentation")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score")

plt.legend()

plt.grid(True)

plt.show()

## Output:

<img width="937" height="418" alt="image" src="https://github.com/user-attachments/assets/a93a508d-8388-42b1-824f-2e455aed14b3" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
