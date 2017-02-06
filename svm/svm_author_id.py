#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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


#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100] 

#features_test = features_test[:len(features_test)/100]
#labels_test = labels_test[:len(labels_test)/100] 

#########################################################
### your code goes here ###
from sklearn import svm

t0 = time()
#clf = svm.SVC(kernel='linear')
clf = svm.SVC(C=10000.0,kernel='rbf')
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()

print len(features_test)
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)


count = 0
for item in pred:
    if item == 1:
        count += 1
print "Chris(1) count: " + str(count)

#########################################################


