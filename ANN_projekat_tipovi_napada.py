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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#scaling of data
scalerX = StandardScaler()
x_train = scalerX.fit_transform(x_train)
x_test = scalerX.transform(x_test)

#making of neural network
ANN = Sequential()
ANN.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu', input_dim = 118))
ANN.add(Dense(output_dim = 23, init = 'uniform', activation = 'softmax'))
ANN.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
ANN.fit(x_train, y_train, batch_size = 10, epochs = 10)

#prediction
y_pred = ANN.predict(x_test)
y_pred = (y_pred > 0.5)

#confusion matrix, accuracy and true positive values
cmatrix = multilabel_confusion_matrix(y_test, y_pred)
print()
print(cmatrix)
for i in range(0, 23):
    print("Preciznost je : " + "za i = " + str(i) + " " + str((cmatrix[i][0][0]+cmatrix[i][1][1])/float(cmatrix[i][0][0]+cmatrix[i][0][1]+cmatrix[i][1][0]+cmatrix[i][1][1])))
    usp = cmatrix[i][1][1]/float(cmatrix[i][1][0]+cmatrix[i][1][1])
    print("Udeo stvarno pozitivnih je : "+ str(usp))
    print()
