#!/usr/bin/python3

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

string_1 = "Hi Katie the self driving car will be late Best Sebastian"
string_2 = "Hi Sebastian the machine learning class will be great great great Best Katie"
string_3 = "Hi Katie the machine learning class will be most excellent"

email_list = [string_1, string_2, string_3]
bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)

print(bag_of_words)
print(vectorizer.vocabulary_.get("great"))
