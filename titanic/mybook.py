# coding: utf-8
import numpy as np
importpandas
import pandas
from sklearn.tree import DecisionTreeClassifier
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data.head()
train_data = np.array([data['Pclass'], data['Fare'], [int(item == 'male') for item in data['Sex']], data['Age']])
train_data[~np.isnan(train_data).any(1)]
train_data = train_data[~np.isnan(train_data).any(1)]
train_data
train_data = np.array([data['Pclass'], data['Fare'], [int(item == 'male') for item in data['Sex']], data['Age']])
train_data
train_data = train_data[~np.isnan(train_data).any(0)]
train_data = train_data[~np.isnan(train_data).any()]
train_data
train_data = np.array([data['Pclass'], data['Fare'], [int(item == 'male') for item in data['Sex']], data['Age']])
train_data
train_data = train_data[~np.isnan(train_data).any(2)]
train_data = train_data[~np.isnan(train_data).any(1)]
train_data
train_data = np.array([data['Pclass'], data['Fare'], [int(item == 'male') for item in data['Sex']], data['Age']])
np.isnan(train_data)
np.isnan(train_data).any(1)
np.isnan(train_data).any(0)
train_data[~np.isnan(train_data).any(0)]
~np.isnan(train_data).any(0)
train_data[~np.isnan(train_data).any(0)]
np.isnan(train_data0
)
np.isnan(train_data)
np.isnan(train_data).any(1)
np.isnan(train_data).any(0)
get_ipython().system(u'np.isnan(train_data).any(0)')
~np.isnan(train_data).any(0)
train_data[~np.isnan(train_data).any(0)]
mask = ~np.isnan(train_data).any(0)
train_data[mask]
train_data[True, False]
train_data[mask]
train_data[np.logical_not(np.isnan(train_data).any(0))]
train_data[np.any(np.isnan(train_data), axis=0)]
train_data[np.any(np.isnan(train_data), axis=1)]
train_data[~np.any(np.isnan(train_data), axis=1)]
train_data[~np.any(np.isnan(train_data), axis=0)]
train_data[~np.any(np.isnan(train_data), axis=2)]
train_data[~np.any(np.isnan(train_data), axis=3)]
train_data[~np.any(np.isnan(train_data), axis=-1)]
train_data[~np.any(np.isnan(train_data))]
train_data[np.any(~np.isnan(train_data), axis=0)]
train_data[np.any(~np.isnan(train_data), axis=1)]
train_data[np.any(~np.isnan(train_data), axis=1)]
train_data[np.any(~np.isnan(train_data), axis=1)]
np.shape(train_data)
np.shape(mask)
np.isnan(train_data)
shape(np.isnan(train_data))
np.shape(np.isnan(train_data))
np.shape(np.isnan(train_data).any(1))
np.shape(np.isnan(train_data).any(0))
np.shape(np.isnan(train_data).any(0))
np.reshape
np.shape
train_data[~np.isnan(train_data).any(0)]
mask = ~np.isnan(train_data.any(0))
np.reshape(mask, (891, 0))
mask = np.array([mask, mask, mask, mask])
train_data[mask]
np.shape(mask)
mask
mask
mask = ~np.isnan(train_data.any(0))
mask
train_data
mask = ~np.isnan(train_data).any(0)
mask = [mask for i in range(4)]
mask
np.shape(mask)
train_data[mask]
train_data[mask]
mask
train_data
train_data[mask]
import numpy as np
x = np.array([[1,2,3], [4,5,np.nan], [7,8,9], [True, False, True]])
print("Original array:")
print(x)
print("Remove all non-numeric elements of the said array")
print(x[~np.isnan(x).any(axis=1)])
import numpy as np
x = np.array([[1,2,3], [4,5,np.nan], [7,8,9], [True, False, True]])
print("Original array:")
print(x)
print("Remove all non-numeric elements of the said array")
print(x[~np.isnan(x).any(axis=0)])
import numpy as np
x = np.array([[1,2,3], [4,5,np.nan], [7,8,9], [True, False, True]])
print("Original array:")
print(x)
print("Remove all non-numeric elements of the said array")
print(np.delete(x, ~np.isnan(x).any(axis=0)))
train_data
mask = ~np.isnan(train_data).any(0)
mask
np.delete(train_data, mask, axis=0)
train_data = np.array([data['Pclass'], data['Fare'], [int(item == 'male') for item in data['Sex']], data['Age']])
np.delete(train_data, mask, axis=1)
np.delete(train_data, mask, axis=2)
np.delete(train_data, mask)
np.delete(train_data, mask, axis=0)
np.shape(np.delete(train_data, mask, axis=0))
mask
np.shape(mask)
test = np.array([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8])
test = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])
test
test = np.array([[1, 2, 3, 4, 5], [2, 3, np.nan, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])
test
~np.isnan(test).any(0)
test[np.isnan(test).any(0)]
data.as_matrix()
data.head()
train_data = np.delete(data.as_matrix(), [0, 2, 5, 6, 7, 9, 10])
train_data
shape(train_data)
np.shape(train_data)
train_data = np.delete(data.as_matrix(), [0, 2, 5, 6, 7, 9, 10], axis=0)
train_data
train_data = np.delete(data.as_matrix(), [0, 2, 5, 6, 7, 9, 10], axis=1)
train_data
for i, item in enumerate(train_data):
    train_data[i][1] = int(item == 'male')
    
enumerate(train_data)
for i, item in enumerate(train_data):
    train_data[i][1] = 0
    
train_data
train_data = np.delete(data.as_matrix(), [0, 2, 5, 6, 7, 9, 10], axis=1)
for i, item in enumerate(train_data):
    train_data[i][1] = item
        
train_data
train_data = np.delete(data.as_matrix(), [0, 2, 5, 6, 7, 9, 10], axis=1)
for i, item in enumerate(train_data):
    train_data[i][1] = (item[1] == 'male')
     
train_data
train_data = np.delete(data.as_matrix(), [0, 2, 5, 6, 7, 9, 10], axis=1)
for i, item in enumerate(train_data):
    train_data[i][1] = int(item[1] == 'male')
    
train_data
train_data = train_data[~np.isnan(train_data).any(1)]
mask
train_data = train_data[~np.isnan(train_data).any(1)]
np.isnan(train_data)
train_data
train_data = np.array(train_data, dtype=np.float32)
train_data
np.isnan(train_data)
train_data = train_data[~np.isnan(train_data).any(1)]
train_data
train_labels = np.array(data['Survived'])
train_data = np.delete(data.as_matrix(), [2, 5, 6, 7, 9, 10], axis=1)
for i, item in enumerate(train_data):
    train_data[i][2] = int(item[1] == 'male')
    
    
train_data
train_data = train_data[~np.isnan(train_data).any(1)]
train_data = np.array(train_data, dtype=np.float32)
train_data = train_data[~np.isnan(train_data).any(1)]
train_data
train_data.hsplit(0)
np.hsplit(train_data)
np.hsplit(train_data, 1)
np.hsplit(train_data, 0)
np.hsplit(train_data, 1)
np.hsplit(train_data, 2)
train_labels = np.array()
train_labels = np.array([])
train_labels.append
train_labels.append(0)
for i, item in enumerate(train_data):
    np.append(train_labels, item[0])
    train_data[i] = train_data[i][1:]
    
train_data[0]
train_data[0][1:]
for i, item in enumerate(train_data):
    np.append(train_labels, item[0])
    train_data[i] = np.delete(item, 0)
    
    
for i, item in enumerate(train_data):
    np.append(train_labels, item[0])
     
train_labels = np.array([])
for item in train_data:
    np.append(train_labels, item[0])
    
train_data = np.delete(train_data, 0, axis=0)
train_data
train_data = np.delete(train_data, 0, axis=1)
train_data
train_data = np.delete(data.as_matrix(), [2, 5, 6, 7, 9, 10], axis=1)
for i, item in enumerate(train_data):
    train_data[i][2] = int(item[1] == 'male')
    
    
    
train_data = train_data[~np.isnan(train_data).any(1)]
train_data = np.array(train_data, dtype=np.float32)
train_data = train_data[~np.isnan(train_data).any(1)]
train_labels = np.array([])
for item in train_data:
    np.append(train_labels, item[0])
    
train_data = np.delete(train_data, 0, axis=1)
train_data
train_labels
train_data = np.delete(data.as_matrix(), [2, 5, 6, 7, 9, 10], axis=1)
for i, item in enumerate(train_data):
    train_data[i][2] = int(item[1] == 'male')
    
    
    
train_data = np.array(train_data, dtype=np.float32)
train_data = train_data[~np.isnan(train_data).any(1)]
train_labels = np.array([])
for item in train_data:
    train_labels = np.append(train_labels, item[0])
    
train_data = np.delete(train_data, 0, axis=1)
train_data
train_labels
clf = DecisionTreeClassifier(random_state=241)
clf.fit(train_data, train_labels)
imp = clf.feature_importances_
imp
get_ipython().magic(u'save')
get_ipython().magic(u'save mybook')
train_data = np.delete(data.as_matrix(), [2, 5, 6, 7, 9, 10], axis=1)
train_data
for i, item in enumerate(train_data):
    train_data[i][2] = int(item[1] == 'male')
    
    
    
train_data
train_data = np.delete(data.as_matrix(), [2, 5, 6, 7, 9, 10], axis=1)
for i, item in enumerate(train_data):
    train_data[i][2] = int(item[2] == 'male')
    
train_data = np.array(train_data, dtype=np.float32)
train_data = train_data[~np.isnan(train_data).any(1)]
train_labels = np.array([])
for item in train_data:
    train_labels = np.append(train_labels, item[0])
    
train_data = np.delete(train_data, 0, axis=1)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(train_data, train_labels)
clf.feature_importances_
train_data
get_ipython().magic(u'save mybook 1-206')
