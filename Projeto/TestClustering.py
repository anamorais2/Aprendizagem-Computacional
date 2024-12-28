import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as shc

# Carregar o dataset
def load_data():
    # Load the COVID_numerics.csv file
    colunas = ["GENDER","AGE","MARITAL STATUS","VACINATION","RESPIRATION CLASS","HEART RATE","SYSTOLIC BLOOD PRESSURE","TEMPERATURE","TARGET"]
    df_numerics = pd.read_csv('COVID_numerics.csv', usecols=colunas)
    # Load the COVID_IMG.csv file without header
    df_img = pd.read_csv('COVID_IMG.csv', header=None)
    return df_numerics, df_img


# Pré-processamento geral
def preprocess_data(df_numerics):
    # Remover duplicados
    df_numerics.drop_duplicates(inplace=True)

    target = df_numerics["TARGET"]
    
    # Excluir a coluna 'TARGET'
    df_numerics = df_numerics.drop(columns=["TARGET"])
    
    # Substituir valores ausentes por média
    df_numerics.fillna(df_numerics.mean(), inplace=True)

    # Remover outliers
    continuous_columns = ["AGE", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE"]
    for column in continuous_columns:
        Q1 = df_numerics[column].quantile(0.25)
        Q3 = df_numerics[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_numerics = df_numerics[(df_numerics[column] >= lower_bound) & (df_numerics[column] <= upper_bound)]
    
    # Normalização
    scaler = MinMaxScaler()
    df_numerics[continuous_columns] = scaler.fit_transform(df_numerics[continuous_columns])
    
    
    return df_numerics, target, df_numerics.index


# Encontrar o número ideal de clusters (K-Means)
def find_optimal_k(data):
    distortions = [] # Soma dos quadrados das distâncias
    silhouette_scores = [] # Coeficiente de Silhouette
    K = range(2, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Plotar o gráfico do cotovelo (Distortion) e Silhouette Scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Método do Cotovelo')

    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores por k')

    plt.tight_layout()
    plt.show()

    return K[np.argmax(silhouette_scores)]

# Aplicar clustering K-Means
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

# Aplicar DBSCAN com busca de parâmetros
def find_best_dbscan_params(data, eps_range, min_samples_range):
    best_eps = None
    best_min_samples = None
    best_silhouette = -1

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            # Ignorar resultados com todos os pontos em um cluster ou como ruído
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_silhouette:
                    best_silhouette = score
                    best_eps = eps
                    best_min_samples = min_samples

    return best_eps, best_min_samples

def apply_agglomerative(data):
    linkage_methods = ['ward', 'complete', 'average', 'single']
    cluster_range = range(2, 11)
    best_linkage = None
    best_clusters = None
    best_silhouette = -1
    best_labels = None

    for linkage in linkage_methods:
        for n_clusters in cluster_range:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = model.fit_predict(data)
            if len(set(labels)) > 1:
                silhouette = silhouette_score(data, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_linkage = linkage
                    best_clusters = n_clusters
                    best_labels = labels

    return best_linkage, best_clusters

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


def find_best_subtractive_params(df, alpha_range, beta_range, radius_range, min_potential_range):
    best_alpha = None
    best_beta = None
    best_radius = None
    best_min_potential = None
    best_silhouette = -1

    for alpha in alpha_range:
        for beta in beta_range:
            for radius in radius_range:
                for min_potential in min_potential_range:
                    centers = subtractive_clustering(df, alpha, beta, radius, min_potential)
                    clusters = assign_clusters(df, centers)
                    score = silhouette_score(df, clusters)
                    if score > best_silhouette:
                        best_silhouette = score
                        best_alpha = alpha
                        best_beta = beta
                        best_radius = radius
                        best_min_potential = min_potential

    return best_alpha, best_beta, best_radius, best_min_potential



# Correlação com regras do domínio e avaliação com Target
def evaluate_clusters(data, labels, original_df, target_column):
    
    if len(original_df) != len(labels):
        raise ValueError("Tamanho dos dados originais não corresponde ao número de clusters gerados.")

    # Adicionar rótulos ao dataset original
    original_df['Cluster'] = labels

    # Correlação com variáveis importantes (regras do domínio)
    print("Correlação com variáveis de domínio:")
    for col in original_df.select_dtypes(include=['float64', 'int64']).columns:
        if col != target_column:
            correlation = original_df.groupby('Cluster')[col].mean()
            print(f"Média de {col} por cluster:\n{correlation}\n")

    # Comparação com o TARGET
    if target_column in original_df.columns:
        cross_tab = pd.crosstab(original_df['Cluster'], original_df[target_column])
        print("\nComparação entre clusters e TARGET:\n")
        print(cross_tab)
        
# Função para plotar os clusters gerados pelo K-Means
def plot_clusters(df, labels, title):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Cluster')
    plt.show()
    
def plot_dendrogram(X, method):
    plt.figure(figsize=(10, 7))
    dend = shc.dendrogram(shc.linkage(X, method))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

    
def SSE(df, centers, clusters):
    distances = np.linalg.norm(df.to_numpy() - centers[clusters], axis=1)
    return np.sum(distances ** 2)

# Pipeline principal
def main():
    # Carregar e pré-processar os dados
    df_numerics, df_img = load_data()
    original_df = df_numerics.copy()
    processed_df, target, valid_indices = preprocess_data(df_numerics)
    
    # Determinar o número ideal de clusters para K-Means
    optimal_k = find_optimal_k(processed_df)
    
    # Aplicar K-Means
    kmeans_labels, kmeans_model = apply_kmeans(processed_df, optimal_k)
    
    plot_clusters(processed_df, kmeans_labels, 'K-Means Clustering')
    
    filtered_df = original_df.loc[valid_indices]

    # Aplicar DBSCAN
    eps_range = np.arange(0.1, 1.0, 0.1)
    min_samples_range = range(2, 10)
    best_eps, best_min_samples = find_best_dbscan_params(processed_df, eps_range, min_samples_range)
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan.fit_predict(processed_df)
    
    plot_clusters(processed_df, dbscan_labels, 'DBSCAN Clustering')
    
    # Aplicar Agglomerative Clustering
    best_linkage, best_clusters  = apply_agglomerative(processed_df.values)
    aggloClustering = AgglomerativeClustering(n_clusters=best_clusters, linkage=best_linkage)
    agglo_labels = aggloClustering.fit_predict(processed_df)
    
    print(f"Best linkage method: {best_linkage}")
    print(f"Best number of clusters: {best_clusters}")
    
    plot_dendrogram(processed_df, best_linkage)   
     
    #Aplicar Subtractive Clustering
    alpha_range = np.arange(1, 5, 0.5)
    beta_range = np.arange(1, 5, 0.5)
    radius_range = np.arange(0.5, 2.0, 0.5)
    min_potential_range = np.logspace(-6, -2, 5)
    best_alpha, best_beta, best_radius, best_min_potential = find_best_subtractive_params(processed_df, alpha_range, beta_range, radius_range, min_potential_range)
    centers = subtractive_clustering(processed_df, alpha=best_alpha, beta=best_beta, radius=best_radius, min_potential=best_min_potential)
    clusters = assign_clusters(processed_df, centers)
    
    print(f"Best alpha: {best_alpha}")
    print(f"Best beta: {best_beta}")
    print(f"Best radius: {best_radius}")
    print(f"Best min_potential: {best_min_potential}")

    plot_clusters(processed_df, clusters, 'Subtractive Clustering')
    
    
    kmeans_silhouette = silhouette_score(processed_df, kmeans_labels)
    dbscan_silhouette = silhouette_score(processed_df, dbscan_labels)
    aggloClustering_silhouette = silhouette_score(processed_df, agglo_labels)
    subtractive_clustering_silhouette = silhouette_score(processed_df, clusters)
    
    print(f"K-Means Silhouette Score: {kmeans_silhouette}")
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
    print(f"Agglomerative Clustering Silhouette Score: {aggloClustering_silhouette}")
    print(f"Subtractive Clustering Silhouette Score: {subtractive_clustering_silhouette}")
    print(f"K-Means SSE: {kmeans_model.inertia_}")
    print("SSE não é aplicável ao DBSCAN, pois não há centroides definidos.")
    print("SSE não é aplicável ao Agglomerative Clustering, pois não há centroides definidos.")
    print(f"Subtractive Clustering SSE: {SSE(processed_df, centers, clusters)}")
   
    # Avaliar clusters
    print("K-Means")
    evaluate_clusters(processed_df, kmeans_labels,filtered_df, 'TARGET')
    print("DBSCAN")
    evaluate_clusters(processed_df, dbscan_labels, filtered_df, 'TARGET')
    print("Agglomerative Clustering")
    evaluate_clusters(processed_df, agglo_labels, filtered_df, 'TARGET')
    print("Subtractive Clustering")
    evaluate_clusters(processed_df, clusters, filtered_df, 'TARGET')

# main_pipeline('dataset.csv', 'TARGET')
if __name__ == '__main__':
    main()
