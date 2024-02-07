import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests

def clustering():
    # Retrieve wine data from API
    response = requests.get('http://localhost:5000/wine-data')
    df = pd.read_json(response.text)
    
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Apply PCA (Principal Component Analysis) for dimensionality reduction
    pca = PCA(n_components=2)  
    data_pca = pca.fit_transform(data_scaled)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Wine_Type'] = kmeans.fit_predict(data_pca)
    print(df)

if __name__ == '__main__':
    clustering()