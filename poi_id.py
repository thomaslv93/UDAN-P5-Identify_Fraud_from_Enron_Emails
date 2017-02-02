#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

## This will be revisited later

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Explore the data
names = data_dict.keys()
all_features = data_dict[names[0]].keys()

num_data_points = len(names)
num_features = len(all_features)

num_pois = 0
num_npois = 0

for name in names:
    if data_dict[name]['poi'] == True:
        num_pois += 1
    elif data_dict[name]['poi'] == False:
        num_npois += 1
        
nancount = {}

for feature in all_features:
    nancount[feature] = 0

for name in names:
    for feature in all_features:
        if data_dict[name][feature] == 'NaN':
            nancount[feature] += 1

print "Number of data points: ", num_data_points
print "Number of poi's: ", num_pois
print "Number of non-poi's: ", num_npois
print "Number of features: ", num_features
print "Missing values for each feature: ", nancount

### Task 2: Remove outliers
import matplotlib
from matplotlib import pyplot

''' for name in names:
    salary = data_dict[name]['salary']
    bonus = data_dict[name]['bonus']
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show() '''

## There is one MASSIVE outlier.

for name in names:
    if data_dict[name]['salary'] > 25000000 and data_dict[name]['salary'] != 'NaN':
        print name

## The outlier is called TOTAL and is represents the aggregate financial data
## which must have been mistakenly entered. We can get rid of this outlier.

del data_dict['TOTAL']
names = data_dict.keys()

''' for name in names:
    salary = data_dict[name]['salary']
    bonus = data_dict[name]['bonus']
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show() '''


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for point in my_dataset:
    my_dataset[point]['poi_email_density'] = 0
    if my_dataset[point]['to_messages'] != 'NaN' and my_dataset[point]['from_messages'] != 'NaN' and my_dataset[point]['from_poi_to_this_person'] != 'NaN' and my_dataset[point]['from_this_person_to_poi'] != 'NaN':
        my_dataset[point]['poi_email_density'] = (my_dataset[point]['from_poi_to_this_person'] + my_dataset[point]['from_this_person_to_poi']) / float(my_dataset[point]['from_messages'] + my_dataset[point]['to_messages'])
        
## Eliminate features with more than 146/2 = 73 NaNs, eliminate email, include poi_email_density
features_list = ['poi',
                 'salary',
                 'to_messages', 
                 'total_payments',
                 'bonus',
                 'total_stock_value', 
                 'shared_receipt_with_poi', 
                 'exercised_stock_options', 
                 'from_messages', 
                 'other', 
                 'from_poi_to_this_person', 
                 'from_this_person_to_poi', 
                 'expenses', 
                 'restricted_stock',
                 'poi_email_density']                 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

kbest = SelectKBest(k = 4)
kbest.fit(features, labels)

final_features = sorted(zip(features_list[1:], kbest.scores_, kbest.get_support()), key = lambda x: x[1])

print "KBest: ", final_features

features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'restricted_stock']
                 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from tester import test_classifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()

test_classifier(clf1, my_dataset, features_list)

from sklearn.tree import DecisionTreeClassifier
parameters = {"criterion": ["gini", "entropy"],
              "splitter": ["best", "random"],
              "max_depth": [None, 2, 4, 8, 16],
              "min_samples_split":  [2, 4, 8, 16],
              "min_samples_leaf": [1, 2, 4, 8, 16],
              "max_leaf_nodes": [None, 4, 8, 16],
              }
dt = DecisionTreeClassifier()
clf2 = GridSearchCV(dt, parameters)
clf2 = clf2.fit(features, labels)

print clf2.best_params_

## {'splitter': 'random', 'max_leaf_nodes': 8, 'min_samples_leaf': 1, 'criterion': 'gini', 'min_samples_split': 4, 'max_depth': None}

clf3 = DecisionTreeClassifier(splitter='random', max_leaf_nodes=8, min_samples_leaf=1, criterion='gini', min_samples_split=4, max_depth=None)

test_classifier(clf3, my_dataset, features_list)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn import metrics
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
print "Accuracy Score: ", metrics.accuracy_score(labels_test, clf.predict(features_test))
print "Precision Score: ", metrics.precision_score(labels_test, clf.predict(features_test))
print "Recall Score: ", metrics.recall_score(labels_test, clf.predict(features_test))

test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)