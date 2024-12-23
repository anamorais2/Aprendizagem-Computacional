# Individuals with suspected COVID are admitted to the hospital emergency room
# At the time of admission, several variables/parameters are acquired (low cost and simple to acquire)
# Based on these variables, the health professional must decide whether the individual remains hospitalized for additional examinations or should return home

# COVID_numerics.csv contains the following variables:

# Screening (1 | Gender; 2 | Age; 3 | Mariatal status; 4 | Vaccinated; 5 | Breathing difficulty)
# Measurements (6 | Heart rate; 7 | Blood pressure) and 8 | Temperature

# knowledge ( If breathing difficulty >= moderate and temperature >= 37.8, then Stay at hospital)

#COVID_IMG.csv contains the following variables:
# ECG - Phase space plot
# Matrix (21,21)
# Binary values {0,1}
# These plot/image can reveal patterns in the ECG data, such as periodicity or anomalies

#  Design a machine learning model to address this issue:
# ▪ Decide whether the individual should remain hospitalized for additional examinations or be discharged to return home.

# The model should be able to:
# ▪ Predict the outcome of the decision based on the variables in the COVID_numerics.csv file and the image in the COVID_IMG.csv file.


# Importing libraries
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Importing the dataset
def load_data():
    # Load the COVID_numerics.csv file
    colunas = ["GENDER","AGE","MARITAL STATUS","VACINATION","RESPIRATION CLASS","HEART RATE","SYSTOLIC BLOOD PRESSURE","TEMPERATURE","TARGET"]
    df_numerics = pd.read_csv('COVID_numerics.csv', usecols=colunas)
    # Load the COVID_IMG.csv file without header
    df_img = pd.read_csv('COVID_IMG.csv', header=None)
    return df_numerics, df_img

# Função de pré-processamento específica para K-Means
def preprocess_KMeans(df_numerics):
    # Remover duplicados
    df_numerics.drop_duplicates(inplace=True)
    
    target = df_numerics["TARGET"]
    
    # Excluir a coluna 'TARGET' do DataFrame, uma vez que não é necessária para o K-Means pois é um modelo não supervisionado
    df_numerics = df_numerics.drop(columns=["TARGET"])
    
    # Substituir valores ausentes por média
    df_numerics.fillna(df_numerics.mean(), inplace=True)

    # Remover outliers para K-Means
    continuous_columns = ["AGE", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE"]
    for column in continuous_columns:
        Q1 = df_numerics[column].quantile(0.25)
        Q3 = df_numerics[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_numerics = df_numerics[(df_numerics[column] >= lower_bound) & (df_numerics[column] <= upper_bound)]
        
    sns.boxplot(data=df_numerics)
    plt.show()

    # Normalização dos dados
    scaler = MinMaxScaler()
    df_numerics[continuous_columns] = scaler.fit_transform(df_numerics[continuous_columns])

    return df_numerics, target

# Função para encontrar o melhor número de clusters
def find_best_k(df, title):
    inertia = []
    silhouette_scores = []
    K = range(2, 11)  # Testar valores de 2 a 10 clusters
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(df, kmeans.labels_)
        silhouette_scores.append(score)
        print(f'For n_clusters = {k}, Silhouette Score = {score}')
    
    # Plotar o método do cotovelo
    plt.figure(figsize=(10, 5))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method For Optimal k ({title})')
    plt.show()
    
    # Plotar a pontuação do coeficiente de silhueta
    plt.figure(figsize=(10, 5))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score For Optimal k ({title})')
    plt.show()

    # Retornar o melhor valor de k baseado na pontuação de silhueta
    best_k = K[silhouette_scores.index(max(silhouette_scores))]
    return best_k

def main():
    # Load the data
    df_numerics, df_img = load_data()
    # First lines of the data numerics
    print(df_numerics.head())
    # Format of the data image
    print(df_img.shape)
    
    # Pré-processamento e clustering para K-Means
    df_numerics_KMeans, target = preprocess_KMeans(df_numerics)
    
    # Plotar a distribuição dos dados
    sns.pairplot(df_numerics_KMeans)
    plt.title('Data Distribution')
    plt.show()
    
    # Encontrar o melhor número de clusters para os dados originais
    best_k_original = find_best_k(df_numerics_KMeans, "Original Data")
    print(f'The best number of clusters for original data is: {best_k_original}')
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numerics_KMeans)
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    print("Variância explicada pelo PCA:", pca.explained_variance_ratio_)
    
    # Plotar a distribuição dos dados após PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=df_pca)
    plt.title('Data Distribution after PCA')
    plt.show()
    
    # Encontrar o melhor número de clusters para os dados com PCA
    best_k_pca = find_best_k(df_pca, "PCA Data")
    print(f'The best number of clusters for PCA data is: {best_k_pca}')
    
    # Clustering com o melhor valor de k para os dados originais
    kmeans_original = KMeans(n_clusters=best_k_original, random_state=42)
    kmeans_original.fit(df_numerics_KMeans)
    df_numerics_KMeans['Cluster'] = kmeans_original.labels_
    
    # Visualizar os clusters para os dados originais
    sns.pairplot(df_numerics_KMeans, hue='Cluster', palette='viridis')
    plt.title('K-Means Clustering with Optimal k (Original Data)')
    plt.show()
    
    # Clustering com o melhor valor de k para os dados com PCA
    kmeans_pca = KMeans(n_clusters=best_k_pca, random_state=42)
    kmeans_pca.fit(df_pca)
    df_pca['Cluster'] = kmeans_pca.labels_
    
    # Visualizar os clusters para os dados com PCA
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis')
    plt.title('K-Means Clustering with Optimal k (PCA Data)')
    plt.show()

if __name__ == '__main__':
    main()