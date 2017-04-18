#!/usr/bin/python3

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../q1_gaussianNB/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()


from sklearn import tree
from sklearn.metrics import accuracy_score

### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(pred, labels_test)
print("DT Accuracy Score: ", acc)


clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
acc_min_samples_split_2 = accuracy_score(pred2, labels_test)


clf50 = tree.DecisionTreeClassifier(min_samples_split=50)
clf50 = clf50.fit(features_train, labels_train)
pred50 = clf50.predict(features_test)
acc_min_samples_split_50 = accuracy_score(pred50, labels_test)

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
#### grader code, do not modify below this line
print(submitAccuracies())
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
