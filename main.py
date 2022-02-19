# Dependencies
import sys

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)
@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['POST'])
def predict():
        try:
            json_ = request.json
            print(json_)
            lr = joblib.load("model.pkl")  # Load "model.pkl"
            print('Model loaded')
            model_columns = joblib.load("model_cols.pkl")  # Load "model_columns.pkl"
            print('Model columns loaded')
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:
            return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':



    app.run(debug=True, port=33507)
