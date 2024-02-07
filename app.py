from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return '¡La aplicación está en funcionamiento!'

@app.route('/wine-data')
def get_wine_data():
    url = 'https://storage.googleapis.com/the_public_bucket/wine-clustering.csv'
    df = pd.read_csv(url)
    return df.to_json()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')