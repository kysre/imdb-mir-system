import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from clustering_metrics import *


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        centroids = {}
        cluster_points = {}
        initial_centroids = [emb_vecs[idx] for idx in np.random.randint(low=0, high=len(emb_vecs), size=n_clusters)]
        for index in range(n_clusters):
            centroids[index] = initial_centroids[index]
            cluster_points[index] = []
        for turn in range(max_iter):
            for index in range(n_clusters):
                cluster_points[index] = []
            for point in emb_vecs:
                distances = []
                for centroid in centroids.values():
                    distances.append(distance.euclidean(point, centroid))
                centroid_group = distances.index(min(distances))
                cluster_points[centroid_group].append(point)
            for k in centroids:
                centroids[k] = np.mean(cluster_points[k], axis=0)
        cluster_labels = []
        for point in emb_vecs:
            distances = []
            for centroid in centroids.values():
                distances.append(distance.euclidean(point, centroid))
            centroid_group = distances.index(min(distances))
            cluster_labels.append(centroid_group)
        return list(centroids.values()), cluster_labels

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        uniques = list(set(documents))
        result = []
        for word in uniques:
            result.append((word, documents.count(word)))
        result = sorted(result, key=lambda x: x[1], reverse=True)
        result = result[:top_n]
        answer = [x[0] for x in result]
        return answer

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        centroids, cluster_labels = self.cluster_kmeans(emb_vecs, n_clusters, max_iter=100)
        WCSS = 0
        for i in range(len(emb_vecs)):
            WCSS += distance.euclidean(emb_vecs[i], centroids[cluster_labels[i]]) ** 2
        return centroids, cluster_labels, WCSS

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='single')
        return clustering.fit_predict(emb_vecs)

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='complete')
        return clustering.fit_predict(emb_vecs)

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='average')
        return clustering.fit_predict(emb_vecs)

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='ward')
        return clustering.fit_predict(emb_vecs)

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, document_labels, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)
        # Perform K-means clustering
        _, labels, _ = self.cluster_kmeans_WCSS(data, n_clusters)
        labels = np.array(labels)
        keys = np.unique(np.array(labels))
        groups = {key: [] for key in range(n_clusters)}
        for i in range(len(data)):
            groups[labels[i]].append(document_labels[i])
        for k in groups:
            groups[k] = self.get_most_frequent_words(groups[k])[0]
        group_document_names = []
        for l in labels:
            group_document_names.append(groups[l])
        group_document_names = np.array(group_document_names)
        colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan', 'tab:grey', 'tab:brown',
                  'tab:orange', 'tab:pink']
        # Plot the clusters
        for k in groups:
            group_data = data[np.argwhere(labels == k).squeeze()]
            plt.scatter(group_data[:, 0], group_data[:, 1], color=colors[k], label=groups[k])
        plt.legend(loc='best')
        # Log the plot to wandb
        wandb.log({"Clustering": wandb.Image(plt)})
        # Close the plot display window if needed (optional)
        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        run = wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(linkage=linkage_method)
        labels = clustering.fit_predict(data)
        # Create linkage matrix for dendrogram
        distances = distance.pdist(data)
        Z = linkage(distances, method=linkage_method)
        dn = dendrogram(Z)
        fig = plt.figure()
        plt.title('Hierarchical Clustering')
        wandb.log({"Hierarchical Clustering": wandb.Image(plt)})
        plt.close()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        metric = ClusteringMetrics()
        for k in k_values:
            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            _, cluster_labels = self.cluster_kmeans(embeddings, k)
            silhouette_scores.append(metric.silhouette_score(embeddings, cluster_labels))
            purity_scores.append(metric.purity_score(true_labels, cluster_labels))

        # Plotting the scores
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(k_values, silhouette_scores)
        plt.title('Silhouette scores')
        plt.subplot(2, 1, 2)
        plt.plot(k_values, purity_scores)
        plt.title('Purity scores')
        # Logging the plot to wandb
        if project_name and run_name:
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Cluster Scores": plt})
        plt.close()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)
        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            _, _, score = self.cluster_kmeans_WCSS(embeddings, k)
            wcss_values.append(score)
        # Plot the elbow method
        plt.plot(k_values, wcss_values, 'bx-')
        plt.title('Elbow Method')
        # Log the plot to wandb
        wandb.log({"Elbow Method": wandb.Image(plt)})
        plt.close()
