#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
### create classifier
clf = GaussianNB()

### fit the classifier on the training features and labels
t0 = time() # start time of fitting
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s") # duration of fitting

### use the trained classifier to predict labels for the test features
t1 = time() # start time of predicting
labels_pred = clf.predict(features_test)
print("prediction time:", round(time()-t1, 3), "s") # duration of preditcting

### calculate and return the accuracy on the test data
### this is slightly different than the example,
### where we just print the accuracy
### you might need to import an sklearn module

accuracy = accuracy_score(labels_test, labels_pred)
print(accuracy)

#########################################################
