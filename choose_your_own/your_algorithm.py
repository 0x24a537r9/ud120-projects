#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]



#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

# K-nearest neighbors ##########################################################
from sklearn.neighbors import KNeighborsClassifier
from time import time

results = []
for n_neighbors in range(1, 10):
  clf = KNeighborsClassifier(n_neighbors=n_neighbors)

  t0 = time()
  clf.fit(features_train, labels_train)
  print 'time: ', round(time() - t0, 3), 's'

  t0 = time()
  labels_pred = clf.predict(features_test)
  print 'time: ', round(time() - t0, 3), 's'

  from sklearn.metrics import accuracy_score
  results.append((n_neighbors, accuracy_score(labels_test, labels_pred)))

print results

# Ada Boost ##########################################################
# from sklearn.ensemble import AdaBoostClassifier
# from time import time

# results = []
# n_estimators_list = [5, 10, 25, 50, 75, 100, 125]
# for n_estimators in n_estimators_list:
#   clf = AdaBoostClassifier(n_estimators=n_estimators)
#   print 'n_estimators: ', n_estimators

#   t0 = time()
#   clf.fit(features_train, labels_train)
#   print 'time: ', round(time() - t0, 3), 's'

#   t0 = time()
#   labels_pred = clf.predict(features_test)
#   print 'time: ', round(time() - t0, 3), 's'

#   from sklearn.metrics import accuracy_score
#   results.append((n_estimators, accuracy_score(labels_test, labels_pred)))

# print results

# Ada Boost ##########################################################
# from sklearn.ensemble import RandomForestClassifier
# from time import time

# results = []
# n_estimators_list = [5, 10, 25, 50, 75, 100, 125]
# for n_estimators in n_estimators_list:
#   clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=40)
#   print 'n_estimators: ', n_estimators

#   t0 = time()
#   clf.fit(features_train, labels_train)
#   print 'time: ', round(time() - t0, 3), 's'

#   t0 = time()
#   labels_pred = clf.predict(features_test)
#   print 'time: ', round(time() - t0, 3), 's'

#   from sklearn.metrics import accuracy_score
#   results.append((n_estimators, accuracy_score(labels_test, labels_pred)))

# print results

try:
    prettyPicture(clf, features_test, labels_test)
    plt.show()
except NameError:
    pass
