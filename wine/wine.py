# coding: utf-8
get_ipython().magic(u'load temp.py')
get_ipython().magic(u'load temp2.py')
# %load temp2.py
import matplotlib.pyplot as plt
help(plt)
import pandas
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
data = pandas.read_csv('wine.csv', header=None)
data.head()
labels = np.array(data[0], dtype=np.int)
params = np.array(np.delete(data.as_matrix(), 0, axis=1), dtype=np.float32)
kf = KFold(5, True, 42)
kf
for i in range(1, 51):
    clf = KNeighborsClassifier(i)
    score = cross_val_score(clf, params, labels, cv=kf)
    print i, score
scores_raw = []
scores_scaled = []
for i in range(1, 51):
    clf = KNeighborsClassifier(i)
    score = cross_val_score(clf, params, labels, cv=kf)
    print i, score
    scores_raw.append(score)
    
for i, score in enumerate(scores_raw):
    print i+1, sum(score)
    
for i, score in enumerate(scores_raw):
    print i+1, sum(score)
    scores_raw[i] = sum(score)
    
scores_raw
max(scores_raw)
max(scores_raw)/4
scale
from sklearn.preprocessing import scale
sparams = scale(params)
sparams
sparams = scale(params)
sparams
for i in range(1, 51):
    clf = KNeighborsClassifier(i)
    score = cross_val_score(clf, sparams, labels, cv=kf)
    print i, score
    scores_scaled.append(score)
    
for i, score in enumerate(scores_scaled):
    print i+1, sum(score)
    
4.8880952381/4
4.8880952381/4
4.8880952381/5
3.6523809523809523/5
