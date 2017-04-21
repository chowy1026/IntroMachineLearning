#!/usr/bin/python3


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

sort_keys = '../tools/python2_lesson14_keys.pkl'
data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
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

### your code goes here

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(labels_test, pred2)
numpy.set_printoptions(precision=2)

print(cnf_matrix)
tn, fp, fn, tp = cnf_matrix.ravel()
print("tn, fp, fn, tp: ", tn, fp, fn, tp)


from sklearn.metrics import classification_report
target_names = ["Non-POI", "POI"]
print(classification_report(labels_test, pred2, target_names=target_names))

from sklearn import metrics
print("precision_score: ", metrics.precision_score(labels_test, pred2))
print("recall_score: ", metrics.recall_score(labels_test, pred2))
print("f1_score", metrics.f1_score(labels_test, pred2))



predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(true_labels, predictions)
numpy.set_printoptions(precision=2)

print(cnf_matrix)
tn, fp, fn, tp = cnf_matrix.ravel()
print("tn, fp, fn, tp: ", tn, fp, fn, tp)



from sklearn.metrics import classification_report
target_names = ["Non-POI", "POI"]
print(classification_report(true_labels, predictions, target_names=target_names))

from sklearn import metrics
print("precision_score: ", metrics.precision_score(true_labels, predictions))
print("recall_score: ", metrics.recall_score(true_labels, predictions))
print("f1_score", metrics.f1_score(true_labels, predictions))
