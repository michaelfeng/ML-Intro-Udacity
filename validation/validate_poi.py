#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn import cross_validation
train_features,test_features,train_labels,test_labels = cross_validation.train_test_split(features,labels)

from sklearn import tree
import numpy as np
clf = tree.DecisionTreeClassifier()

### it's all yours from here forward!
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print clf.score(features,labels)
# 0.98958333333333337

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels,test_size=0.3,random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print clf.score(features_test,labels_test)
# 0.72413793103448276

preds = clf.predict(features_test)
num_pois = 0

for pred in preds:
    if pred == 1:
        num_pois += 1

print "Number of POIs predicted:", num_pois
print "Total number of people in test set:", len(preds)

true_positives = 0
false_positives = 0
false_negatives = 0

for pred, actual in zip(preds, labels_test):
    if pred == actual and actual == 1:
        true_positives += 1
    elif pred == 1 and actual == 0:
        false_positives += 1
    elif pred == 0 and actual == 1:
        false_negatives += 1

print "True Positives:", true_positives
print "False Positives", false_positives
print "False Negatives", false_negatives

print "\nPrecision (POIs):", true_positives/float(true_positives + false_positives)
print "Recall (POIs)", true_positives/float(true_positives + false_negatives)
