#!/usr/bin/python

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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print len(enron_data)

count = 0
sample = {}
for item in enron_data:
    if enron_data[item]["poi"] ==1:
        count += 1
    sample = enron_data[item]    

print sample.keys()
        
print count
print enron_data['PRENTICE JAMES']['salary'] 
print enron_data['PRENTICE JAMES']['total_stock_value'] 


print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print '##'
print enron_data['SKILLING JEFFREY K']['total_payments']

print enron_data['LAY KENNETH L']['total_payments']

print enron_data['FASTOW ANDREW S']['total_payments']

print '##'
count_salary = 0
count_email = 0
for key in enron_data.keys():
    if enron_data[key]['salary'] != 'NaN':
        count_salary+=1
    if enron_data[key]['email_address'] != 'NaN':
        count_email+=1
print count_salary
print count_email

print '##'
count_NaN_tp = 0
for key in enron_data.keys():
    if enron_data[key]['total_payments'] == 'NaN':
        count_NaN_tp+=1

print count_NaN_tp
print float(count_NaN_tp)/len(enron_data.keys())

print '##'
count_NaN_tp = 0
for key in enron_data.keys():
    if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi'] == True :
        print
        count_NaN_tp+=1
print count_NaN_tp
print float(count_NaN_tp)/len(enron_data.keys())

print '##'
count = 0
for user in enron_data:
    if enron_data[user]['poi'] == True and enron_data[user]['total_payments'] == 'NaN':
        count+=1
print count
