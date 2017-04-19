#!/usr/bin/python3


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

sort_keys = '../tools/python2_lesson13_keys.pkl'
data = featureFormat(data_dict, features_list, sort_keys = sort_keys)
labels, features = targetFeatureSplit(data)

import numpy
numpy.random.seed(42)

from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!
from sklearn import tree
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

pred = clf.predict(features)
acc = accuracy_score(pred, labels)
print("DT Accuracy Score: ", acc)

print(clf.feature_importances_)

clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(features_train, labels_train)

pred2 = clf2.predict(features_test)
acc2 = accuracy_score(pred2, labels_test)
print("DT2 Accuracy Score: ", acc2)
print(clf2.feature_importances_)
