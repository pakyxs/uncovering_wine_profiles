from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wine-data')
def get_wine_data():
    url = 'https://storage.googleapis.com/the_public_bucket/wine-clustering.csv'
    df = pd.read_csv(url)
    return df.to_json()

@app.route('/clustering')
def clustering():

    wine_data_json = get_wine_data()
    
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
    return df.to_json()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    clustering()