#!/usr/bin/python3

from sklearn.linear_model import Lasso

features, labels = getMyData()
regression = Lasso()
regression.fit(features, labels)
regression.predict([2,4])
print(regression.coef_)

# if the print output/return [7.0, 0.0], that means only one feature that is really important.  The one with the coefficient of 7.0.  The one with 0.0 are less important. 
