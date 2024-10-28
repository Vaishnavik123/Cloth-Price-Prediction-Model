import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
encoder=pickle.load(open("encoder.pkl","rb"))

@flask_app.route("/")
def Home():
    return render_template("index (2).html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    input_dict = {
        'type': [request.form['type']],
        'brand': [request.form['brand']],
        'material': [request.form['material']],
        'style': [request.form['style']],
        'color': [request.form['color']],
        'state': [request.form['state']]
    }
    inputs_df = pd.DataFrame(input_dict)
    encoded_cols = encoder.get_feature_names_out(['type','brand','material','style','color','state']).tolist()
    inputs_df[encoded_cols] = encoder.transform(inputs_df)
    predictions=model.predict(inputs_df[encoded_cols])
    price=float(predictions[0])
    price=round(price,2)
    prediction_text = f"<strong style='font-size: 1.5em;'>The price is ${price}</strong>"

    return render_template("index (2).html", prediction_text=prediction_text)


if __name__ == "__main__":
    flask_app.run(debug=True)
