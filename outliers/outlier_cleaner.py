#!/usr/bin/python3

import pprint
import numpy
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    err = (predictions - net_worths) ** 2
    cleaned_data = zip(ages, net_worths, err)
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2][0], reverse=True)
    limit = int(len(net_worths)*0.1)
    #
    # pprint.pprint(numpy.sort(diff, axis = None))
    # print(type(predictions))
    # print(type(net_worths))
    ### your code goes here


    return cleaned_data[limit:]
