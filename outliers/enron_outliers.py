#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
# print [(key, value['salary']) for key, value in data_dict.items() if value['salary'] != 'NaN']
print sorted([(key, value['salary']) for key, value in data_dict.items()
              if value['salary'] != 'NaN'], key=lambda k: -k[1])[0:5]
print [(key, value['salary']) for key, value in data_dict.items()
       if value['salary'] != 'NaN' and value['salary'] > 1000000 and value['bonus'] > 5000000]


for point in data:
  salary = point[0]
  bonus = point[1]
  matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
