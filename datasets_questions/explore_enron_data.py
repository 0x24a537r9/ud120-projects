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

print 'People: ', len(enron_data)
print 'Features: ', len(enron_data.values()[0])
print 'Example features: ', enron_data.values()[0]
print 'POIs: ', sum(person['poi'] for person in enron_data.values())
print 'James\'s stock: ', enron_data['PRENTICE JAMES']['total_stock_value']
print 'Wesley\'s emails to POI: ', enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print 'Jeffrey\'s exercised_stock_options: ', enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print 'Jeffrey\'s total pay: ', enron_data['SKILLING JEFFREY K']['total_payments']
print 'Kenneth\'s total pay: ', enron_data['LAY KENNETH L']['total_payments']
print 'Andy\'s total pay: ', enron_data['FASTOW ANDREW S']['total_payments']

print 'People w/ salary: ', sum(1 for person in enron_data.values() if person['salary'] != 'NaN')
print 'People w/ email: ', sum(1 for person in enron_data.values() if person['email_address'] != 'NaN')
print 'People w/o total payment: ', sum(1 for person in enron_data.values() if person['total_payments'] == 'NaN')
print 'POIs w/o total payment: ', sum(1 for person in enron_data.values()
                                      if person['total_payments'] == 'NaN' and
                                      person['poi'] == 1)
