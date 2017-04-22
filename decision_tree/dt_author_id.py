#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

from sklearn import tree
print len(features_train[0])

classifier = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
classifier.fit(features_train, labels_train)
print 'time: ', round(time() - t0, 3), 's'

t0 = time()
labels_pred = classifier.predict(features_test)
print 'time: ', round(time() - t0, 3), 's'

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, labels_pred)
print [labels_pred[10], labels_pred[26], labels_pred[50]]

print sum(labels_pred)
