# <h1 align=center> **Wine type prediction using machine learning** </h1>

This repository contains a Dockerized analysis pipeline for clustering wine data retrieved via an API call using Flask.

## Overview
- Data Description
- Usage
- Endpoints
- Files
- Dependencies
- Credits

## Description of Dataset
If you download the dataset, you can see that several features will be used to classify the quality of wine, many of them are chemical, so we need to have a basic understanding of such chemicals.

- Alcohol: In wine, alcohol refers to ethanol, which is produced through the fermentation of sugars by yeast. It contributes to the body, texture, and mouthfeel of the wine, as well as its perceived sweetness and warmth.

- Malic acid: Malic acid is one of the primary acids found in wine grapes. It contributes to the tartness and sourness of the wine. During fermentation, malic acid can be converted into lactic acid through malolactic fermentation, which can soften the acidity of the wine.

- Ash: Ash refers to the inorganic residue left behind after the organic matter in wine is burned off. It consists mainly of minerals such as potassium, calcium, and magnesium, which are essential for yeast metabolism and can influence wine stability and flavor.

- Ash Alkalinity: Ash alkalinity is a measure of the buffering capacity of wine, which indicates its ability to resist changes in pH when acids or bases are added. It is influenced by the presence of alkaline substances such as potassium and calcium.

- Magnesium: Magnesium is one of the minerals found in wine ash. It plays a role in yeast metabolism during fermentation and can also affect the sensory characteristics of wine.

- Total Phenols: Total phenols represent the sum of all phenolic compounds present in wine, including flavonoids and non-flavonoid phenols. These compounds contribute to the color, flavor, and mouthfeel of wine, as well as its antioxidant properties.

- Flavonoids: Flavonoids are a subgroup of phenolic compounds found in wine, including flavonols, flavan-3-ols, anthocyanins, and flavones. They contribute to the color, flavor, and antioxidant activity of wine.

- Non-flavonoid Phenols: Non-flavonoid phenols are another subgroup of phenolic compounds found in wine, including phenolic acids, stilbenes, and tannins. They also contribute to the color, flavor, and antioxidant properties of wine.

- Proanthocyanins: Proanthocyanins are a type of flavonoid found in wine, particularly in red wines, where they contribute to color stability, mouthfeel, and astringency.

- Color Intensity: Color intensity refers to the depth or concentration of color in wine, which is influenced by factors such as grape variety, winemaking techniques, and aging.

- HUE: HUE is a measure of the color tint or hue of wine, indicating its dominant color tones such as red, purple, or yellow.

- OD280: OD280, or Optical Density at 280 nm, is a measure of the absorbance of light at a specific wavelength by phenolic compounds in wine. It is used to assess the concentration of phenolic compounds, which can indicate wine quality and aging potential.

- Proline: Proline is an amino acid found in grapes and wine. It contributes to the flavor and structure of wine and is often used as an indicator of grape ripeness and fermentation conditions.

### Usage

Follow the instructions below to run the analysis pipeline:

1. **Clone the Repository**:

    ```bash
    git clone uncovering_wine_profiles
    ```

2. **Build the Docker Image**:

    ```bash
    docker build -t wine-clustering .
    ```

3. **Run the Docker Container**:

    ```bash
    docker run -p 5000:5000 wine-clustering
    ```

### Endpoints

1. **Access Wine Data**:

    To access raw DataFrame, copy on your browser:
    ```bash
    127.0.0.1:5000/wine-data
    ```

2. **Access Clustering DataFrame Results**:

    To access DataFrame with the type of wine identified, copy on your browser:
    ```bash
    127.0.0.1:5000/clustering
    ```

3. **Access Elbow Method Visual**:

    To access Elbow Method visual to identify the optimal number of cluster, copy on your browser:
    ```bash
    127.0.0.1:5000/visual-before-pca
    ```

4. **Access Clustering Visual Before PCA**:

    To access Clustering Visual Before PCA, copy on your browser:
    ```bash
    127.0.0.1:5000/visual-before-pca
    ```
5. **Access Clustering Visual After PCA**:

    To access Clustering Visual After PCA, copy on your browser:
    ```bash
    127.0.0.1:5000/visual-after-pca
    ```

### Files

- `notebooks`: Contains notebooks with the data analysis and ML model.
- `templates`: Contains HTML template for the endpoints.
- `Dockerfile`: Contains instructions for building the Docker image.
- `requirements.txt`: Lists the dependencies required for the analysis.
- `app.py`: Flask API for serving the wine dataset.
- `README.md`: Instructions for running the analysis pipeline.

### Dependencies

- Flask
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Requests
- Docker

### Credits

- This analysis pipeline was created by Moreira Rodrigo.
