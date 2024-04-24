import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt


def readCSV(filepath):
    df = pd.read_csv(filepath, engine='pyarrow')
    return df


def preprocessData(data_frame):
    # Identify numeric columns
    numeric_columns = data_frame.select_dtypes(include=[np.number]).columns.tolist()

    # If you want to include categorical data, identify categorical columns
    categorical_columns = data_frame.select_dtypes(exclude=[np.number]).columns.tolist()
    # Create a transformer for one-hot encoding categorical data
    transformer = ColumnTransformer([
        ('scale', StandardScaler(), numeric_columns),
        ('one_hot', OneHotEncoder(), categorical_columns)
    ],
        remainder='passthrough',  # Pass through other columns unchanged
        sparse_threshold=0  # Ensures the output is a dense array
    )

    transformed_data = transformer.fit_transform(data_frame)

    one_hot_features = transformer.named_transformers_['one_hot'].get_feature_names_out(categorical_columns)
    features = np.append(numeric_columns, one_hot_features)

    transformed_df = pd.DataFrame(transformed_data, columns=features)

    return transformed_df


def KmeansClustering(data_frame):
    feature_names = data_frame.columns.tolist()
    clustered = cluster.KMeans(n_clusters=5)
    clustered.fit(
        data_frame[feature_names]
    )
    return clustered.labels_, clustered.cluster_centers_


if __name__ == "__main__":
    filepath = 'mushroom.csv'
    DF = readCSV(filepath)
    DF = preprocessData(DF)
    labels, center = KmeansClustering(DF)

    plt.figure(figsize=(10,10))
    plt.scatter(DF.iloc[:, 0], DF.iloc[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5, label='clusters')
    plt.scatter(center[:, 0], center[:, 1], s=300, c='red', marker='x', label='Centroids')
    plt.xlabel(DF.columns[0])
    plt.ylabel(DF.columns[1])
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()


