import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import random



def import_dataset():
    print("Reading table")
    return pd.read_csv('data/mall.csv')


def change_size(percentage_to_keep):
    dataset = import_dataset()
    # Calculate the number of rows to keep based on the percentage
    num_rows_to_keep = int(len(dataset) * (percentage_to_keep / 100))
    # Randomly select the rows to keep
    rows_to_keep = random.sample(range(len(dataset)), num_rows_to_keep)
    # Create a new data frame with only the selected rows
    return dataset.iloc[rows_to_keep]


def read_values():
    print("Reading table")
    return dataset.iloc[:, [3, 4]].values


def fillWcss():
    print("Filling wcss")
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)


def visualize(perc):
    print("Visualizing wcss")
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method for csv size ' + str(perc))
    plt.xlabel('Number of clusters')
    plt.ylabel('WCC')
    plt.show()


def vis_clusters(x, y_kmeans, kmeans, perc):
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of customers for cluster size ' + str(perc))
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for value in [10, 20, 35, 50, 75, 100]:
        wcss = []
        dataset = change_size(value)
        x = read_values()
        fillWcss()
        visualize(value)

        print("Belonging to a cluster")
        kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(x)
        print(y_kmeans)

        vis_clusters(x, y_kmeans, kmeans, value)
