#!/usr/bin/python3

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
# clf = SVC(kernel="linear")
#
# ### The following two lines cuts down the training data to 1/100 of the dataset.
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
#
# #### now your job is to fit the classifier
# #### using the training features/labels, and to
# #### make a set of predictions on the test data
# t0 = time() # start time of fitting
# clf.fit(features_train, labels_train)
# print("training time: ", round(time()-t0, 3), "s") # duration of fitting
#
#
# #### store your predictions in a list named pred
# t1 = time() # start time of predicting
# pred = clf.predict(features_test)
# print("prediction time: ", round(time()-t1, 3), "s") # duration of preditcting
#
from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)
# print("Accuracy: ", acc)

#########################################################



def rbf_fit(C=1.0, gamma='auto'):
    print("==== rbf fitting with C ", str(C), " and gamma ", str(gamma), " ====")
    from sklearn.svm import SVC
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

    from sklearn.metrics import accuracy_score
    rbf_acc = accuracy_score(pred, labels_test)
    print("rbf accuracy: ", rbf_acc)

    print("=========================  end fitting  ============================")

# list_C = [1.0, 10.0, 100.0, 1000.0, 10000.0]
# for c in list_C:
#     rbf_fit(c)


def my_rbf_fit():
    print("====================== rbf fitting with C = 10000  ======================")
    clf = SVC(kernel="rbf", C=10000.0)

    #### Using 1% of the training dataset to speed up
    # features_train = features_train[:int(len(features_train)/100)]
    # labels_train = labels_train[:int(len(labels_train)/100)]
    #### now your job is to fit the classifier
    #### using the training features/labels, and to
    #### make a set of predictions on the test data
    t0 = time() # start time of fitting
    clf.fit(features_train, labels_train)
    print("rbf training time: ", round(time()-t0, 3), "s") # duration of fitting

    #
    #### store your predictions in a list named pred
    t1 = time() # start time of predicting
    pred = clf.predict(features_test)
    print("rbf prediction time: ", round(time()-t1, 3), "s") # duration of preditcting

    # print(type(pred))
    print("Count of Predicted Chris: ", (pred == 1).sum())
    print("Count of Predicted Sara: ", (pred == 0).sum())

    # print("predictions[10] ", pred[10])
    # print("predictions[26] ", pred[26])
    # print("predictions[50] ", pred[50])

    rbf_acc = accuracy_score(pred, labels_test)
    print("rbf accuracy: ", rbf_acc)



    print("=========================  end fitting  ============================")

print("full_features_train", len(features_train))
print("full_features_test", len(features_test))
print("full_labels_train", len(labels_train))
print("full_labels_test", len(labels_test))

my_rbf_fit()
