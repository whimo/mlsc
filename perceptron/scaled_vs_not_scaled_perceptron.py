# coding: utf-8
import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
train = pandas.read_csv('perceptron-train.csv')
train = pandas.read_csv('perceptron-train.csv', header=None)
test = pandas.read_csv('perceptron-test.csv', header=None)
train.head()
test.head()
train_X = train.as_matrix([1, 2])
train_y = train.as_matrix([0])
train_X
test_X = test.as_matrix([1, 2])
test_y = test.as_matrix([0])
clf = Perceptron(random_state=241)
clf.fit(train_X, train_y)
train_y
train_y = data[0]
train_y = train[0]
train
train_y
type(train_y)
train_y = np.array(train[0])
test_y = np.array(test[0])
train_y
clf.fit(train_X, train_y)
prediction = clf.predict(test_X)
prediction
orig_score = accuracy_score(test_y, prediction)
orig_score
scaler = StandardScaler()
scaled_train_X = scaler.fit_transform(train_X)
scaled_test_X = scaler.transform(test_X)
sclf = Perceptron(random_state=241)
sclf.fit(scaled_train_X, train_y)
scaled_prediction = sclf.predict(scaled_test_X)
scaled_score = accuracy_score(test_y, prediction)
scaled_score
scaled_score = accuracy_score(test_y, scaled_prediction)
scaled_score
