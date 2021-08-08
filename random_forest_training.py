import time
import pickle

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


df = pd.read_csv('data/4class_categorised.csv')
df.drop('Unnamed: 0', axis='columns', inplace=True)

X = df[['followers','friends','favorites','mentions','hashtags','urls','sentistrength']]
X = sm.add_constant(X).values
Y = df['retweet_group'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

start_time = time.time()
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
end_time = time.time()
print('Time to train model:', end_time - start_time)

# save
with open('data/random_forest.pkl','wb') as f:
    pickle.dump(clf,f)

start_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
print('Time to predict:', end_time - start_time)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, digits=3))