from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
import base64

app = Flask(__name__)


# Flask route to display index
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to display the wine data
@app.route('/wine-data')
def get_wine_data():
    url = 'https://storage.googleapis.com/the_public_bucket/wine-clustering.csv'
    df = pd.read_csv(url)
    return df.to_json()


# Flask route to display the DataFrame with clustering performed
@app.route('/clustering')
def clustering():

    wine_data_json = get_wine_data()
    
    # Transform JSON datato DataFrame
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
    
    # Print DataFrame
    return df.to_json()

# Function to display elbow method
def elbow_method():

    wine_data_json = get_wine_data()

    # Transform JSON datato DataFrame
    df = pd.read_json(wine_data_json)
    
    # Elbow method
    X_cluster = StandardScaler().fit_transform(df)
    # Finds best number of clusters using Inertia_ metric
    SSE_data = []
    for n in range(2, 20):
        # Perform the clustering
        kmeans = KMeans(n_clusters=n)
        model = kmeans.fit(X_cluster)
        SSE_data.append(model.inertia_)

    # Plot the SSE values to find the elbow that gives the best number of clusters
    SSE_series = pd.Series(SSE_data)
    x = np.arange(2., 20., 1.0)
    plt.scatter(x, SSE_series, c="b", marker='o', label="SSE vs. n_clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.legend(loc=1)
    plt.ylim(ymin=1)
    plt.grid()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the image data as base64
    img_base64 = base64.b64encode(img.getvalue()).decode()

    return img_base64

# Flask route to display the elbow method
@app.route('/elbow-method')
def show_visualization():
    img = elbow_method()
    return render_template('elbow-method.html', img=img)


# Function to display clustering before PCA
def before_pca():
    wine_data_json = get_wine_data()

    # Transform JSON datato DataFrame
    df = pd.read_json(wine_data_json)

    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Choose the number of clusters (you may need domain knowledge or use techniques like the elbow method to determine this)
    num_clusters = 3

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Wine_Type'] = kmeans.fit_predict(data_scaled)

    # Visualize the clusters (assuming the data has only two features)
    plt.scatter(df['Flavanoids'], df['Hue'], c=df['Wine_Type'], cmap='viridis')
    plt.xlabel('Flavonoids')
    plt.ylabel('HUE')
    plt.title('K-means Clustering before PCA')
    plt.show()

    # Save the plot to a BytesIO object
    img_before_pca = io.BytesIO()
    plt.savefig(img_before_pca, format='png')
    img_before_pca.seek(0)

    # Encode the image data as base64
    img_before_pca_base64 = base64.b64encode(img_before_pca.getvalue()).decode()

    return img_before_pca_base64
    



# Flask route to display clustering before PCA
@app.route('/visual-before-pca')
def show_before_pca():
    img = before_pca()
    return render_template('visual-before-pca.html', img=img)

# Function to display clustering after PCA
def after_pca():

    wine_data_json = get_wine_data()

    # Transform JSON datato DataFrame
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

    # Visualize the clusters
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=df['Wine_Type'], cmap='viridis')
    plt.xlabel('Flavonoids')
    plt.ylabel('HUE')
    plt.title('K-means Clustering after PCA')
    plt.show()

    # Save the plot to a BytesIO object
    img_after_pca = io.BytesIO()
    plt.savefig(img_after_pca, format='png')
    img_after_pca.seek(0)

    # Encode the image data as base64
    img_after_pca_base64 = base64.b64encode(img_after_pca.getvalue()).decode()

    return img_after_pca_base64

# Flask route to display clustering after CPA
@app.route('/visual-after-pca')
def show_after_pca():
    img = after_pca()
    return render_template('visual-after-pca.html', img=img)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    clustering()
