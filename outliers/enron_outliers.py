#!/usr/bin/python3

import pickle
import sys
import matplotlib.pyplot
import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)
pprint.pprint(data)
# sorted_data = sorted(data,key=lambda x:x[2][0], reverse=True)

### your code below
# ax = matplotlib.pyplot.subplots()

fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111)

for point in data:
    salary = point[0]
    bonus = point[1]
    print(type(point))
    matplotlib.pyplot.scatter( salary, bonus )

    ax.annotate('(%s, %s)' % (point[1], point[0]), xy=point, textcoords='data') # <--


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
