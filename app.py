from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/wine-data')
def get_wine_data():
    url = 'https://storage.googleapis.com/the_public_bucket/wine-clustering.csv'
    df = pd.read_csv(url)
    return df.to_json()

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1')