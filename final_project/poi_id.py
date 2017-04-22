#!/usr/bin/python

from matplotlib import pyplot
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from time import time
import itertools
import math
import numpy as np
import pickle
import sys

sys.path.append('../tools/')
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


################################################################################
### CONSTANTS ##################################################################
################################################################################


FINANCE_FEATURE_NAMES = [
  'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
  'exercised_stock_options', 'expenses', 'loan_advances', 'long_term_incentive',
  'other', 'restricted_stock', 'restricted_stock_deferred', 'salary',
  'total_payments', 'total_stock_value',
]

EMAIL_FEATURE_NAMES = [
  'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
  'shared_receipt_with_poi', 'to_messages',
]

COMPUTED_FEATURE_NAMES = [
  'from_this_person_to_poi_percent',
  'from_poi_to_this_person_pct'
]

ALL_RAW_FEATURE_NAMES = ['poi'] + FINANCE_FEATURE_NAMES + EMAIL_FEATURE_NAMES
ALL_FEATURE_NAMES = ALL_RAW_FEATURE_NAMES + COMPUTED_FEATURE_NAMES

FOLDS = 100


################################################################################
### HELPER FUNCTIONS ###########################################################
################################################################################


def show_feature_scatter_plot_matrix(data_dict, sampled_features, filename):
  data = featureFormat(data_dict, sampled_features, remove_all_zeroes=True,
                       sort_keys=True)

  fig, axes = pyplot.subplots(nrows=data.shape[1], ncols=data.shape[1],
                              figsize=(8, 8))
  fig.subplots_adjust(hspace=0.05, wspace=0.05)
  poi_feature_dists = data[(1==data[:,0])].T
  non_poi_feature_dists = data[(0==data[:,0])].T
  for ax in axes.flat:
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if ax.is_first_col():
        ax.yaxis.set_ticks_position('left')
    if ax.is_last_col():
        ax.yaxis.set_ticks_position('right')
    if ax.is_first_row():
        ax.xaxis.set_ticks_position('top')
    if ax.is_last_row():
        ax.xaxis.set_ticks_position('bottom')

  for i, j in zip(*np.triu_indices_from(axes, k=1)):
    for x, y in [(i, j), (j, i)]:
      axes[x, y].scatter(non_poi_feature_dists[x], non_poi_feature_dists[y],
                         marker='.', s=5, alpha=.7, edgecolors='face', c='c')
      axes[x, y].scatter(poi_feature_dists[x], poi_feature_dists[y], marker='.',
                         s=5, alpha=.7, edgecolors='face', c='r')

  for i, label in enumerate(sampled_features):
    axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                        ha='center', va='center')

  for i, j in zip(range(len(sampled_features)), itertools.cycle((-1, 0))):
    axes[j, i].xaxis.set_visible(True)
    axes[i, j].yaxis.set_visible(True)

  fig.suptitle('Feature Scatter Plot Matrix')
  pyplot.savefig(filename)
  pyplot.show()


def show_features_scatter_plot(data_dict, sampled_features):
  sampled_features = sampled_features[0:2]
  data = featureFormat(
      data_dict,
      ['poi'] + sampled_features,
      remove_all_zeroes=True,
      sort_keys=True)

  pyplot.xlabel(sampled_features[0])
  pyplot.ylabel(sampled_features[1])

  poi_feature_dists = data[(1==data[:,0])].T
  non_poi_feature_dists = data[(0==data[:,0])].T
  
  pyplot.scatter(non_poi_feature_dists[1], non_poi_feature_dists[2],
                 marker='.', s=10, alpha=.7, edgecolors='face', c='c')
  pyplot.scatter(poi_feature_dists[1], poi_feature_dists[2], marker='.',
                 s=10, alpha=.7, edgecolors='face', c='r')

  pyplot.savefig('_'.join(sampled_features) + '.pdf')
  pyplot.show()


################################################################################
### LOADING / PREFILTERING #####################################################
################################################################################


# Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'r') as data_file:
  data_dict = pickle.load(data_file)

# Filter out bad data points early so it doesn't affect our analyses below.
del data_dict['TOTAL']  # Invalid data point.
del data_dict['THE TRAVEL AGENCY IN THE PARK']  # Non-person.

featureless_points = [
    name for name, features in data_dict.iteritems()
    if all(f == 'NaN' or f == 0 for f in features.values())
]
for name in featureless_points:
  del data_dict[name]


################################################################################
### FILTERING / TRANSFORMATION #################################################
################################################################################


# Compute some additional features.
for features in data_dict.values():
  try:
    features['from_this_person_to_poi_percent'] = (
        features['from_this_person_to_poi'] /
        float(features['from_messages']))
  except:
    features['from_this_person_to_poi_percent'] = 'NaN'

  try:
    features['from_poi_to_this_person_pct'] = (
        features['from_poi_to_this_person'] /
        float(features['to_messages']))
  except:
    features['from_poi_to_this_person_pct'] = 'NaN'

if '--corr' in sys.argv:
  show_features_scatter_plot(data_dict,
      ['from_this_person_to_poi_percent', 'from_poi_to_this_person_pct'])

# Convert the features and labels to numpy arrays and scale.
scaler = MinMaxScaler()

labels, finance_features = targetFeatureSplit(featureFormat(
    data_dict, ['poi'] + FINANCE_FEATURE_NAMES, sort_keys=True,
    remove_all_zeroes=False))
finance_features = scaler.fit_transform(finance_features)

# TODO: PCA on the finance features.
# pca = PCA(n_components=2, whiten=True).fit(X_train)
# X_training_pca = pca.transform(X_train)
# training_X_pca = pca.transform(training_X)
# print 'first PC variance: ', pca.explained_variance_ratio_[0]
# print 'second PC variance: ', pca.explained_variance_ratio_[1]

_, email_features = targetFeatureSplit(featureFormat(
    data_dict, ['poi'] + EMAIL_FEATURE_NAMES, sort_keys=True,
    remove_all_zeroes=False))
email_features = scaler.fit_transform(email_features)  # Probably not necessary.

_, computed_features = targetFeatureSplit(featureFormat(
    data_dict, ['poi'] + COMPUTED_FEATURE_NAMES, sort_keys=True,
    remove_all_zeroes=False))
computed_features = scaler.fit_transform(computed_features)

# TODO: Best % feature selection.
# sp = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=30)
# all_features = sp.fit_transform(all_features, labels)

all_labels = np.asarray(labels)
all_features = np.concatenate(
    (finance_features, email_features, computed_features), axis=1)


################################################################################
### ANALYSIS ###################################################################
################################################################################


if '--basic_stats' in sys.argv:
  names = sorted(set(data_dict.keys()))
  print('\n# of raw features: %s' % len(ALL_RAW_FEATURE_NAMES))
  print('# of available points: %s' % len(names))
  print('Available points: %s\n' % ', '.join(names))

if '--feature_stats' in sys.argv:
  # Calculate the number/percent each feature is defined in the dataset.
  feature_population = []
  for feature in ALL_FEATURE_NAMES:
    count = sum(1 for point in data_dict.values() if point[feature] != 'NaN')
    feature_population.append((feature, count, count / float(len(data_dict))))
  feature_population = sorted(feature_population, key=lambda x: -x[2])
  print('Feature population count and %% (in descending order):\n%s\n' %
        '\n'.join('  %s - %s (%d%%)' % (feature, count, round(100 * percent))
                  for feature, count, percent in feature_population))

if '--feature_corr_matrix' in sys.argv:
  show_feature_scatter_plot_matrix(
      data_dict, ['poi'] + FINANCE_FEATURE_NAMES, 'finance_features.pdf')
  show_feature_scatter_plot_matrix(
      data_dict, ['poi'] + EMAIL_FEATURE_NAMES, 'email_features.pdf')
  show_feature_scatter_plot_matrix(
      data_dict, ['poi'] + COMPUTED_FEATURE_NAMES, 'computed_features.pdf')

if '--feature_corrs' in sys.argv:
  show_features_scatter_plot(data_dict, ['poi', 'salary'])
  show_features_scatter_plot(data_dict, ['shared_receipt_with_poi', 'expenses'])


################################################################################
### TRAINING ###################################################################
################################################################################


print('Fitting the classifier to the training set...')

precision_scores = []
recall_scores = []
cross_validator = StratifiedShuffleSplit(n_splits=FOLDS)
for training_i, testing_i in cross_validator.split(all_features, all_labels):
  training_features = all_features[training_i]
  training_labels = all_labels[training_i]
  testing_features = all_features[testing_i]
  testing_labels = all_labels[testing_i]
  # print training_labels
  # print testing_labels

  # SVM
  # param_grid = {
  #   'C': [1e-1, 1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
  #   'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
  # }
  # clf = GridSearchCV(SVC(), param_grid)

  # Naive-Bayes
  # param_grid = {}
  # clf = GridSearchCV(GaussianNB(), param_grid)

  # Random Forest
  from sklearn.ensemble import RandomForestClassifier
  param_grid = {
    'n_estimators': [1, 2, 3, 5, 7, 10],
    'min_samples_split': range(20, 30, 3),
  }
  clf = GridSearchCV(RandomForestClassifier(), param_grid)

  # Decision Tree
  # from sklearn.tree import DecisionTreeClassifier
  # param_grid = {
  #   'min_samples_split': range(20, 30),
  # }
  # clf = GridSearchCV(DecisionTreeClassifier(presort=True), param_grid)

  # AdaBoost
  # from sklearn.ensemble import AdaBoostClassifier
  # param_grid = {
  #     'n_estimators': [1, 2, 3, 5, 7, 10],
  # }
  # clf = GridSearchCV(AdaBoostClassifier(), param_grid)

  clf = clf.fit(training_features, training_labels)
  # print clf.best_estimator_

  testing_pred = clf.predict(testing_features)
  precision_scores.append(precision_score(testing_labels, testing_pred))
  recall_scores.append(recall_score(testing_labels, testing_pred))
  print classification_report(testing_labels, testing_pred, target_names=['non-poi', 'poi'])
  print confusion_matrix(testing_labels, testing_pred, labels=range(2))


################################################################################
### EVALUATION #################################################################
################################################################################


print 'precision_scores: ', precision_scores
print 'recall_scores: ', recall_scores
print 'average precision: ', sum(precision_scores) / float(FOLDS)
print 'average recall: ', sum(recall_scores) / float(FOLDS)


################################################################################
### OUTPUT #####################################################################
################################################################################


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, data_dict, ALL_FEATURE_NAMES)