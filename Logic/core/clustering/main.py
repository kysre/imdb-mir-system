import os
import json

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dimension_reduction import DimensionReduction
from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks
BASE_PATH = '/Users/divar/University/term-8/information-retrieval/imdb-mir-system/Logic/data/clustering'
PROJECT_NAME = 'clustering'
RUN_NAME = 'phase_2'
K_VALUES = list(range(2, 9))

if __name__ == '__main__':
    # 0. Embedding Extraction
    embeddings = np.load(f'{BASE_PATH}/embeddings.npy')
    labels = np.load(f'{BASE_PATH}/labels.npy')
    with open(f'{BASE_PATH}/document_labels.json', 'r') as f:
        document_labels = json.loads(f.read())
        f.close()

    # 1. Dimension Reduction
    # Perform Principal Component Analysis (PCA):
    dimension_reduction = DimensionReduction()
    pca_embed = dimension_reduction.pca_reduce_dimension(embeddings, 2)
    dimension_reduction.wandb_plot_explained_variance_by_components(embeddings, PROJECT_NAME, RUN_NAME)
    # Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
    tsne_embeddings = dimension_reduction.convert_to_2d_tsne(embeddings)
    dimension_reduction.wandb_plot_2d_tsne(embeddings, PROJECT_NAME, RUN_NAME)

    # 2. Clustering
    # K-Means Clustering
    clustering = ClusteringUtils()
    metrics = ClusteringMetrics()
    clustering.visualize_elbow_method_wcss(tsne_embeddings, K_VALUES, PROJECT_NAME, RUN_NAME)
    clustering.plot_kmeans_cluster_scores(tsne_embeddings, labels, K_VALUES, PROJECT_NAME, RUN_NAME)
    for k in K_VALUES:
        clustering.visualize_kmeans_clustering_wandb(tsne_embeddings, k, document_labels, PROJECT_NAME, RUN_NAME)
    # Hierarchical Clustering
    for linkage_method in ['single', 'average', 'complete', 'ward']:
        clustering.wandb_plot_hierarchical_clustering_dendrogram(tsne_embeddings, PROJECT_NAME, linkage_method,
                                                                 RUN_NAME)

    # 3. Evaluation
    complete_pred = clustering.cluster_hierarchical_complete(tsne_embeddings)
    average_pred = clustering.cluster_hierarchical_average(tsne_embeddings)
    single_pred = clustering.cluster_hierarchical_single(tsne_embeddings)
    ward_pred = clustering.cluster_hierarchical_ward(tsne_embeddings)
    for method, pred in [
        ('complete', complete_pred),
        ('average', average_pred),
        ('single', single_pred),
        ('ward', ward_pred),
    ]:
        print(f'test result on {method} linkage method')
        print(f'purity score: {metrics.purity_score(labels, pred)}')
        print(f'silhouette score: {metrics.silhouette_score(tsne_embeddings, pred)}')
        print(f'adjusted rand score: {metrics.adjusted_rand_score(labels, pred)}')
