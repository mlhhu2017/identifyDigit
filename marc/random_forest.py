import mnist_util
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import Isomap
from sklearn import tree
from sklearn import metrics
import numpy as np
import pydot
import subprocess
from io import StringIO

train_x, train_y, test_x, test_y = mnist_util.get_np_array()
train_x, train_y = shuffle(train_x, train_y, random_state=0)
test_x, test_y = shuffle(test_x, test_y, random_state=0)

NUM_TRAIN_SAMPLES = 10000

print('RANDOM FOREST CLASSIFIER')

NUM_TEST_SAMPLES = int(NUM_TRAIN_SAMPLES*0.20) # 80/20
if NUM_TEST_SAMPLES >= test_x.shape[0]:
    print('Correcting test split')
    NUM_TEST_SAMPLES = test_x.shape[0]

print('Fitting RandomForest')
clf = RandomForestClassifier(n_estimators=100, criterion="gini", n_jobs=4)
clf.fit(train_x[:NUM_TRAIN_SAMPLES], train_y[:NUM_TRAIN_SAMPLES])

print('Calculating Accuracy')
pred_test = clf.predict(test_x[:NUM_TEST_SAMPLES])
acc = metrics.classification_report(test_y[:NUM_TEST_SAMPLES].tolist(), pred_test, digits=4)
print(acc)

print('----------------------------------------------------')

print('DECISION TREE CLASSIFIER')

print('Fitting Decision Tree')
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=train_x.shape[1])
clf.fit(train_x[:NUM_TRAIN_SAMPLES], train_y[:NUM_TRAIN_SAMPLES])

print('Calculating Accuracy')
pred_test = clf.predict(test_x[:NUM_TEST_SAMPLES])
acc = metrics.classification_report(test_y[:NUM_TEST_SAMPLES].tolist(), pred_test, digits=4)
print(acc)

plot_q = input('Do you want to export the  Decision Tree Dot Graph? (Y/n): ')
if plot_q.lower() == 'y':
    with open("dt.dot", 'w') as f:
        tree.export_graphviz(clf, out_file=f)
    print('Exported the dot file')

print('----------------------------------------------------')