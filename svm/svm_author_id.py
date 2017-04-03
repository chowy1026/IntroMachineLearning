#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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
from sklearn.svm import SVC
clf = SVC(kernel="linear")

### The following two lines cuts down the training data to 1/100 of the dataset.
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
t0 = time() # start time of fitting
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s") # duration of fitting


#### store your predictions in a list named pred
t1 = time() # start time of predicting
pred = clf.predict(features_test)
print("prediction time: ", round(time()-t1, 3), "s") # duration of preditcting


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuracy: ", acc)

#########################################################



def rbf_fit(C=1.0, gamma='auto'):
    print("==== rbf fitting with C ", str(C), " and gamma ", str(gamma), " ====")
    clf = SVC(kernel="rbf", C=C, gamma=gamma)
    #### now your job is to fit the classifier
    #### using the training features/labels, and to
    #### make a set of predictions on the test data
    t0 = time() # start time of fitting
    clf.fit(features_train, labels_train)
    print("rbf training time: ", round(time()-t0, 3), "s") # duration of fitting


    #### store your predictions in a list named pred
    t1 = time() # start time of predicting
    pred = clf.predict(features_test)
    print("rbf prediction time: ", round(time()-t1, 3), "s") # duration of preditcting

    rbf_acc = accuracy_score(pred, labels_test)
    print("rbf accuracy: ", rbf_acc)

    print("=========================  end fitting  ============================")

list_C = [1.0, 10.0, 100.0, 1000.0, 10000.0]
for c in list_C:
    rbf_fit(c)
