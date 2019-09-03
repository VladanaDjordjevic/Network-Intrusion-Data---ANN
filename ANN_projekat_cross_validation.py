from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from math import log
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)

#loading of data
information_base = pd.read_csv('kddcup.data_10_percent_corrected_proba.txt')
information_base = information_base.drop_duplicates()

#preprocessing of data, transformation of categorical values
x = information_base.iloc[:, :-1].values
y = information_base.iloc[:, 41].values

LEncoderX1 = LabelEncoder()
LEncoderX2 = LabelEncoder()
LEncoderX3 = LabelEncoder()
x[:, 1] = LEncoderX1.fit_transform(x[:, 1])
x[:, 2] = LEncoderX1.fit_transform(x[:, 2])
x[:, 3] = LEncoderX1.fit_transform(x[:, 3])

OHEncoder1 = OneHotEncoder(categorical_features = [1])
x = OHEncoder1.fit_transform(x).toarray()
OHEncoder2 = OneHotEncoder(categorical_features = [4])
x = OHEncoder2.fit_transform(x).toarray()
OHEncoder3 = OneHotEncoder(categorical_features = [70])
x = OHEncoder3.fit_transform(x).toarray()

LEncoderY = LabelEncoder()
y = LEncoderY.fit_transform(y)

OHEncoder4 = OneHotEncoder(categorical_features = [0])
y = y.reshape(-1, 1)
y = OHEncoder4.fit_transform(y).toarray()

#names of classes and numbers after transformation
classes = LEncoderY.classes_
classes = list(classes)
print(classes)
print(LEncoderY.transform(classes))

#method for cross validation split
kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=seed)
cvscores = []
#making of neural network with cross validation
true_positive = [0 for i in range(23)]
for train, test in kfold.split(x, np.zeros(shape=(y.shape[0], 1))):
    
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    
    #scaling of data
    scalerX = StandardScaler()
    x_train = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)
    
    ANN = Sequential()
    ANN.add(Dense(activation="relu", input_dim=118, units=30, kernel_initializer="uniform"))
    ANN.add(Dense(activation="softmax", units=23, kernel_initializer="uniform"))
    ANN.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    ANN.fit(x_train, y_train, batch_size = 5, epochs = 10)
    scores = ANN.evaluate(x_test, y_test, verbose=0)
    
    print("%s: %.2f%%" % (ANN.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    #counting of true positive values
    y_pred = ANN.predict(x_test)
    y_pred = (y_pred > 0.5)
    cmatrix = multilabel_confusion_matrix(y_test, y_pred)
    for i in range(0,23):
        k = cmatrix[i][1][1]/float(cmatrix[i][1][0]+cmatrix[i][1][1])
        if k == k:
            true_positive[i] += k

#mean of accuracy and true positive values
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
array = [x/23 for x in true_positive]
print(array)
