import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def readCSV(filepath):
    df = pd.read_csv(filepath, engine='pyarrow')
    return df

def preprocessData(data_frame):
    numeric_columns = data_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data_frame.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer = ColumnTransformer([
        ('scale', StandardScaler(), numeric_columns),
        ('one_hot', OneHotEncoder(), categorical_columns)
    ], remainder='passthrough', sparse_threshold=0)
    transformed_data = transformer.fit_transform(data_frame)
    one_hot_features = transformer.named_transformers_['one_hot'].get_feature_names_out(categorical_columns)
    features = np.append(numeric_columns, one_hot_features)
    transformed_df = pd.DataFrame(transformed_data, columns=features)
    return transformed_df

def applyPCA(data_frame, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_frame)
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components+1)])
    return principal_df

def KmeansClustering(data_frame, n_clusters=5):
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(data_frame)
    return kmeans.labels_, kmeans.cluster_centers_

if __name__ == "__main__":
    filepath = 'mushroom.csv'
    DF = readCSV(filepath)
    DF = preprocessData(DF)
    PCA_DF = applyPCA(DF)
    labels, centers = KmeansClustering(PCA_DF)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(PCA_DF['PC1'], PCA_DF['PC2'], c=labels, cmap='viridis', alpha=0.5, marker='o', label='Data Points')
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='x', label='Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-means Clustering on PCA-reduced Data')
    plt.legend()
    plt.colorbar(scatter)
    plt.show()
