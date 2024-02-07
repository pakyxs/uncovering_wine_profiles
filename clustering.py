import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests

def clustering():
    # Hacer la solicitud HTTP a la ruta /wine-data
    response = requests.get('http://127.0.0.1:5000/wine-data')
    
    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        # Leer los datos JSON de la respuesta
        wine_data_json = response.json()
        
        # Convertir los datos JSON a un DataFrame
        df = pd.read_json(wine_data_json)
        
        # Standardize the features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)

        # Apply PCA (Principal Component Analysis) for dimensionality reduction
        pca = PCA(n_components=2)  
        data_pca = pca.fit_transform(data_scaled)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Wine_Type'] = kmeans.fit_predict(data_pca)
        
        # Imprimir el DataFrame después de realizar el clustering
        print(df)
    else:
        print("Error: No se pudo obtener los datos de vino.")

if __name__ == '__main__':
    clustering()