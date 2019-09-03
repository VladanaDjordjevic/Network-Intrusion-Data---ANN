import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
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

#transformation of the target class by replacing attack names with "attack"
information_base['normal.'] = information_base['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

#preprocessing of data, transformation of categorical values
x = information_base.iloc[:, :-1].values
y = information_base.iloc[:, 41].values

LEncoderX1 = LabelEncoder()
LEncoderX2 = LabelEncoder()
LEncoderX3 = LabelEncoder()
x[:, 1] = LEncoderX1.fit_transform(x[:, 1])
x[:, 2] = LEncoderX2.fit_transform(x[:, 2])
x[:, 3] = LEncoderX3.fit_transform(x[:, 3])

OHEncoder1 = OneHotEncoder(categorical_features = [1])
x = OHEncoder1.fit_transform(x).toarray()
OHEncoder2 = OneHotEncoder(categorical_features = [4])
x = OHEncoder2.fit_transform(x).toarray()
OHEncoder3 = OneHotEncoder(categorical_features = [70])
x = OHEncoder3.fit_transform(x).toarray()

LEncoderY = LabelEncoder()
y = LEncoderY.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#scaling of data
scalerX = StandardScaler()
x_train = scalerX.fit_transform(x_train)
x_test = scalerX.transform(x_test)

#making of neural network
ANN = Sequential()
ANN.add(Dense(output_dim = 30, init = 'uniform', activation = 'sigmoid', input_dim = 118))
ANN.add(Dense(output_dim = 30, init = 'uniform', activation = 'sigmoid'))
ANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ANN.fit(x_train, y_train, batch_size = 10, nb_epoch = 10)

#prediction
y_pred = ANN.predict(x_test)
y_pred = (y_pred > 0.5)

#confusion matrix, accuracy and true positive values
cmatrix = confusion_matrix(y_test, y_pred)
print (cmatrix)
print("Preciznost: "+ str((cmatrix[0][0]+cmatrix[1][1])/float(cmatrix[0][0]+cmatrix[0][1]+cmatrix[1][0]+cmatrix[1][1])))
print("Udeo lazno pozitivnih: "+ str(cmatrix[1][0]/float(cmatrix[0][0]+cmatrix[1][0])))
precision = cmatrix[1][1]/float(cmatrix[1][0]+cmatrix[1][1])
print("Udeo stvarno pozitivnih: "+ str(precision))
print("Entropija je "+ str(-precision*log(precision)))
