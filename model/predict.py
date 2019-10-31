import numpy as np
from PIL import Image
import base64
import io
import pickle
import joblib
import statistics

def convert(data):
    data=base64.decodebytes(bytes((data[22:]),'utf-8'))
    image = Image.open(io.BytesIO(data))
    #converting the image to greyscale
    image=image.convert('L')
    #resizing with smoothing (ANTIALIAS)
    image=image.resize((28,28),Image.ANTIALIAS)
    #converting the image to array
    image = np.asarray(image)
    return image.flatten()

def myPrediction(digit, classifier):
    testImage = []
    testImage.append(digit)
    testImage = np.asarray(testImage)
    return classifier.predict(testImage)

#Loading the saved trained models
def loadTrainModels():
    svmClassifier = joblib.load('model/trainedModel/svmClassifier.pickle')
    knnClassifier = joblib.load('model/trainedModel/knnClassifier.pickle').set_params(n_jobs=1)
    rfcClassifier = joblib.load('model/trainedModel/rfcClassifier.pickle').set_params(n_jobs=1)
    return [svmClassifier, knnClassifier, rfcClassifier]

def predict(imageURL):
    digit = convert(imageURL)
    prediction = []
    classifiers = loadTrainModels()
    svmClassifier = classifiers[0]
    knnClassifier = classifiers[1]
    rfcClassifier = classifiers[2]
    prediction.append(myPrediction(digit, svmClassifier)[0])
    prediction.append(myPrediction(digit, knnClassifier)[0])
    prediction.append(myPrediction(digit, rfcClassifier)[0])
    return (statistics.mode(prediction))