from flask import Flask, render_template, request
app = Flask(__name__)
from model.predict import predict
# from model.preprocessing.preprocessing import preprocessing
from model.predict import predict
import numpy as np
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def digitPrediction():
    imageURL = (request.form.get('imageURL',False))
    digitPrediction = predict(imageURL)
    print(digitPrediction)
    return str(digitPrediction)

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)


#Entry and exit point of our flask application