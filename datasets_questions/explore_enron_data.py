#!/usr/bin/python3

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

with open("../final_project/final_project_dataset.pkl", "rb") as f:
    enron_data = pickle.load(f)

print("Enron Data: ", enron_data)
print("Number of Data Points: ", str(len(enron_data)))
print("Number of Features: ", str(len(enron_data["SKILLING JEFFREY K"])))

poi_count=0
for k, v in enron_data.items():
    if v['poi'] == True:
        poi_count += 1
print("Number of POI: ", str(poi_count))
print("Our number of POI: ", str(35))
print("PRENTICE JAMES 's stocks ", str(enron_data['PRENTICE JAMES']['total_stock_value']))
print("COLWELL WESLEY 's to POI ", str(enron_data['COLWELL WESLEY']['from_this_person_to_poi']))
print("Exercised Stock by SKILLING JEFFREY K ", str(enron_data['SKILLING JEFFREY K']['exercised_stock_options']))

#'FASTOW ANDREW S', 'LAY KENNETH L', 'SKILLING JEFFREY K'
payouts = {'FASTOW ANDREW S':enron_data['FASTOW ANDREW S']['total_payments'], 'LAY KENNETH L': enron_data['LAY KENNETH L']['total_payments'], 'SKILLING JEFFREY K': enron_data['SKILLING JEFFREY K']['total_payments']}
print("%s got the biggest payout of %s" % (max(payouts, key=payouts.get), str(max(payouts.values()))))



salary_count=0
for k, v in enron_data.items():
    if v['salary'] != 'NaN':
        salary_count += 1
print("Number with Quantifiable Salary: ", str(salary_count))

email_count=0
for k, v in enron_data.items():
    if v['email_address'] != 'NaN':
        email_count += 1
print("Number with Known Email: ", str(email_count))

tot_payment_nan_count=0
for k, v in enron_data.items():
    if v['total_payments'] == 'NaN':
        tot_payment_nan_count += 1
print("Number with NaN Total Payment: ", str(tot_payment_nan_count))
print("Percentage of Data Points with NaN Total Payment: ", str(float(tot_payment_nan_count)/float(len(enron_data))))

poi_tot_payment_nan_count=0
for k, v in enron_data.items():
    if v['total_payments'] == 'NaN' and v['poi'] == True:
        poi_tot_payment_nan_count += 1
print("Number with POI NaN Total Payment: ", str(poi_tot_payment_nan_count))
print("Percentage of Data Points with POI NaN Total Payment: ", str(float(poi_tot_payment_nan_count)/float(len(enron_data))))

hypo_tot_payment_nan_count = tot_payment_nan_count + 10
hypo_poi = poi_count + 10
hyto_enron_data_length = len(enron_data) + 10
hypo_poi_tot_payment_nan_count = poi_tot_payment_nan_count + 10
print("HYPO NaN Total Payment Count: ", str(hypo_tot_payment_nan_count))
print("HYPO POI Count: ", str(hypo_poi))
print("HYPO Enron Data Point Count: ", str(hyto_enron_data_length))
print("HYPO POI NaN Total Payment Count: ", str(hypo_poi_tot_payment_nan_count))
