#!/usr/bin/python

from time import time
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
#plt.show()
################################################################################

from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs

t0 = time()
''' Extremely Random Forest 
clf = RandomForestClassifier(n_estimators=10, max_depth=None,
                                 min_samples_split=2, random_state=0)
X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
                      random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean()
'''

''' Random Forest - 92%
clf = RandomForestClassifier(n_estimators=10)
'''

''' AdaBoost - 92.4%
iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
scores.mean()
'''

''' SVM - 93.2%'''
clf = svm.SVC(gamma=6, C=10.0,kernel='rbf')


''' KNN - 92.8%
n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
'''
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary





'''
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
'''
