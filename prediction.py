import time
import pickle
import numpy as np
# from numpy import load

from sklearn import metrics

# load array
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# load
with open('data/random_forest.pkl', 'rb') as f:
    clf = pickle.load(f)

start_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
print('Time to predict:', end_time - start_time)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, digits=3))