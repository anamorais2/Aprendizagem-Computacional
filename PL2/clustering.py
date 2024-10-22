import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score


def load_data(nome_ficheiro):
    df = pd.read_csv(nome_ficheiro, header=None)
    df.columns = ['x', 'y']
    return df


def kmeans(df, k, max_iters=20):
    kmeans = KMeans(n_clusters=k, max_iter=max_iters)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_
    clusters = kmeans.labels_
    return centroids, clusters

def agglomerative_clustering(df, k):
    ac = AgglomerativeClustering(n_clusters=k) 
    ac.fit(df) 
    labels = ac.labels_
    return labels


def dbscan(df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)
    return labels

# Função para Subtractive Clustering
def subtractive_clustering(df, alpha=0.5, beta=0.8, radius=1.0, min_potential=1e-5):
    df = df.to_numpy()
    n_points = df.shape[0]
    potentials = np.zeros(n_points)

    # Calculando os potenciais iniciais para cada ponto
    for i in range(n_points):
        distances = np.linalg.norm(df[i] - df, axis=1)
        potentials[i] = np.sum(np.exp(-alpha * (distances**2)))

    centers = []
    while True:
        # Encontrar o ponto com maior potencial
        max_potential_idx = np.argmax(potentials)
        max_potential = potentials[max_potential_idx]

        if max_potential < min_potential:
            break

        # Armazenar o centro do cluster
        centers.append(df[max_potential_idx])

        # Reduzir os potenciais dos pontos próximos
        distances = np.linalg.norm(df - df[max_potential_idx], axis=1)
        potentials -= max_potential * np.exp(-beta * (distances**2))

    return np.array(centers)


def assign_clusters(df, centers):
    df = df.to_numpy()
    distances = np.linalg.norm(df[:, None] - centers, axis=2)
    return np.argmin(distances, axis=1)

# Função para avaliação (Soma dos quadrados dos erros - SSE)
def evaluation(df, centers, clusters):
    distances = np.linalg.norm(df.to_numpy() - centers[clusters], axis=1)
    return np.sum(distances ** 2)


def plot_clusters(df, centroids, clusters, title):
    plt.scatter(df['x'], df['y'], c=clusters, s=5, cmap='viridis')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=50, c='red', marker='X')
    plt.title(title)
    plt.show()

def plot_dendrogram(df):
    dendrogram(linkage(df, method='ward'))
    plt.title('Dendrograma')
    plt.show()


def test_kmeans_agglomerative(df, k_range):
    best_k_kmeans = None
    best_silhouette_kmeans = -1
    best_k_agg = None
    best_silhouette_agg = -1

    for k in k_range:
        # KMeans
        centroids, clusters = kmeans(df, k)
        silhouette_kmeans = silhouette_score(df, clusters)
        sse_kmeans = evaluation(df, centroids, clusters)
        print(f"KMeans (K={k}): SSE = {sse_kmeans}, Silhouette = {silhouette_kmeans}")
        if silhouette_kmeans > best_silhouette_kmeans:
            best_silhouette_kmeans = silhouette_kmeans
            best_k_kmeans = k

        # Agglomerative Clustering
        labels_agg = agglomerative_clustering(df, k)
        silhouette_agg = silhouette_score(df, labels_agg)
        print(f"Agglomerative Clustering (K={k}): Silhouette = {silhouette_agg}")
        if silhouette_agg > best_silhouette_agg:
            best_silhouette_agg = silhouette_agg
            best_k_agg = k

    print(f"\nMelhor K para KMeans: {best_k_kmeans} com Silhouette Score: {best_silhouette_kmeans} com SSE: {sse_kmeans}")
    print(f"Melhor K para Agglomerative: {best_k_agg} com Silhouette Score: {best_silhouette_agg} com SSE: {sse_kmeans}")

# Função principal
if __name__ == "__main__":
    df = load_data("P2_CLUSTER6.csv") 

    # Subtractive Clustering
    centers = subtractive_clustering(df, alpha=4, beta=1.5)
    cluster = assign_clusters(df, centers)
    plot_clusters(df, centers, cluster, "Subtractive Clustering")

    # KMeans Clustering - Teste para diferentes K
    k_range = range(2, 10)
    test_kmeans_agglomerative(df, k_range)
    
    #plot de Dendrograma
    plot_dendrogram(df)
    #plot de KMeans
    centroids, clusters = kmeans(df, 2)
    plot_clusters(df, centroids, clusters, "KMeans Clustering" + " K=2")
    
    # DBSCAN Clustering
    labels_dbscan = dbscan(df, eps=4, min_samples=5)
    plot_clusters(df, None, labels_dbscan, "DBSCAN Clustering")

    # Avaliação Subtractive Clustering
    evaluation_subtractive = evaluation(df, centers, cluster)
    print(f"Subtractive SSE: {evaluation_subtractive}")
    print(f"Subtractive Silhouette Score: {silhouette_score(df, cluster)}")
    

    # Avaliação DBSCAN
    print(f"DBSCAN: {len(set(labels_dbscan))} clusters")
    if len(set(labels_dbscan)) > 1:  # Verifica se há mais de um cluster
        silhouette_score_dbscan = silhouette_score(df, labels_dbscan)
        print(f"DBSCAN Silhouette Score: {silhouette_score_dbscan}")
    else:
        print("DBSCAN não conseguiu encontrar clusters")
