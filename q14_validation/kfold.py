#!/usr/bin/python3

import pickle
import numpy
numpy.random.seed(42)

from sklearn.metrics import accuracy_score




### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



from time import time
t0 = time()

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB

kf = KFold(n_splits=2)
### sklearn v0.16
# cv = KFold( len(authors), 2 ) # not shuffled
# cv = KFold( len(authors), 2, shuffle=True ) # to shuffle data
### sklearn v0.18.1
# cv = KFold( n_splits=2 ) # not shuffled
# cv = KFold( n_splits=2, shuffle=True )

for train, test in kf.split(word_data):
    print("%s %s" % (train, test))
    features_train = [word_data[idx] for idx in train]
    features_test = [word_data[idx] for idx in test]
    labels_train = [authors[idx] for idx in train]
    labels_test = [authors[idx] for idx in test]

    # features_train, features_test, labels_train, labels_test = word_data[train], word_data[test], authors[train], authors[test]


    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test).toarray()


    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()


    clf = GaussianNB()
    clf = clf.fit(features_train_transformed, labels_train)
    print("training time:", round(time()-t0, 3), "s")

    t1 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t1, 3), "s") # duration of preditcting


    acc = accuracy_score(pred, labels_test)
    print("Accuracy Score: ", acc)
