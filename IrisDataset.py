#%%
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import numpy as np
from IPython.display import display

iris_dataset = load_iris()
print("Keys of iris dataset: \n{}".format(iris_dataset.keys()))
#%%
print(iris_dataset['DESCR'][:193] + '\n...')
#%%
print("Target Name: {}".format(iris_dataset['target_names']))
#%%
print("Feature Names: {}".format(iris_dataset['feature_names']))
#%%
print("Type of data: {}".format(type(iris_dataset['data'])))
#%%
print("Shape of data: {}".format(iris_dataset['data'].shape))
#%%
print("First five rows of data: \n{}".format(iris_dataset['data'][:5]))
#%%
print("Type of Target: {}".format(type(iris_dataset['target'])))
#%%
print("Shape of Target: {}".format(iris_dataset['target'].shape))
#%%
print("Target: \n{}".format(iris_dataset['target']))
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("Y_test shape: {}".format(Y_test.shape))
#%%
iris_dataFrame = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataFrame, c=Y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
#%% md
# Building Model: K-Nearest Neighbour (KNN)
#%%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, Y_train)
#%%
x_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(x_new.shape))
#%%
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Predicted Target Name: {}".format(iris_dataset['target_names'][prediction]))
#%%
y_pred = knn.predict(X_test)
print("Test set Prediction: \n{}".format(y_pred))
#%%
print("Test set score: {:.2f}".format(np.mean(y_pred == Y_test)))
#%%
print("Test set score: {:.2f}".format(knn.score(X_test, Y_test)))