#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import sys
import numpy as np
import pickle
from sklearn import model_selection, svm, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import joblib 
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')


# %%


# Load MNIST Data
print('\nLoading MNIST Data...')
data = MNIST('./MNIST_Dataset_Loader/dataset/')
print('\nLoaded MNIST Data...')


# %%


print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)
print('\nLoaded Training Data...')


# %%


print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)
print('\nLoaded Testing Data...')


# %%


#test_labels[0]


# %%

"""
for index in range(train_img.shape[0]):
    if train_labels[index] == 9:
        plt.imshow(train_img[index].reshape(28,28),cmap='gray')
        plt.show()
"""

# %%


def svmClassifier(train_img, train_labels, test_img, test_labels):
    #Features
    X = train_img
    #Labels
    Y = train_labels
    #Fitting on Training Data
    svmClassifier = svm.SVC(gamma=0.1, kernel='poly')
    svmClassifier.fit(X, Y)
    #Predicting on Testing Data
    test_labels_pred = svmClassifier.predict(test_img)
    #Calculating accuracy
    acc = accuracy_score(test_labels, test_labels_pred)
    #Calculating confusion matrix
    conf_mat = confusion_matrix(test_labels, test_labels_pred)
    # Save the model as a pickle in a file 
    joblib.dump(svmClassifier, 'svmClassifier.pickle') 
    print('Accuracy is \n', acc)
    print('Confusion Matrix is \n', conf_mat)


# %%


def knnClassifier(train_img, train_labels, test_img, test_labels):
    #Features
    X = train_img
    #Labels
    Y = train_labels
    #Fitting on Training Data
    knnClassifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=10)
    knnClassifier.fit(X, Y)
    #Predicting on Testing Data
    test_labels_pred = knnClassifier.predict(test_img)
    #Calculating accuracy
    acc = accuracy_score(test_labels, test_labels_pred)
    #Calculating confusion matrix
    conf_mat = confusion_matrix(test_labels, test_labels_pred)
    # Save the model as a pickle in a file 
    joblib.dump(knnClassifier, 'knnClassifier.pickle') 
    print('Accuracy is \n', acc)
    print('Confusion Matrix is \n', conf_mat)


# %%


def rfcClassifier(train_img, train_labels, test_img, test_labels):
    #Features
    X = train_img
    #Labels
    Y = train_labels
    #Fitting on Training Data
    rfcClassifier = RandomForestClassifier(n_estimators=100, n_jobs=10)
    rfcClassifier.fit(X, Y)
    #Predicting on Testing Data
    test_labels_pred = rfcClassifier.predict(test_img)
    #Calculating accuracy
    acc = accuracy_score(test_labels, test_labels_pred)
    #Calculating confusion matrix
    conf_mat = confusion_matrix(test_labels, test_labels_pred)
    # Save the model as a pickle in a file 
    joblib.dump(rfcClassifier, 'rfcClassifier.pickle') 
    print('Accuracy is \n', acc)
    print('Confusion Matrix is \n', conf_mat)


# %%

print('\n Training SVM Classifier \n')
svmClassifier(train_img, train_labels, test_img, test_labels)
print('\n Completed SVM Classifier training \n')

# %%

print('\n Training KNN Classifier \n')
knnClassifier(train_img, train_labels, test_img, test_labels)
print('\n Completed KNN CLassifier training \n')

# %%

print('\n Training RFC Classifier \n')
rfcClassifier(train_img, train_labels, test_img, test_labels)
print('\n Completed RFC Clasifier training \n')


# %%


# %%
