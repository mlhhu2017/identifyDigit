import datetime as dt
import numpy as np

from sklearn.metrics import confusion_matrix as conf_mat
from mnist_util import *

np.set_printoptions(threshold=np.nan)

PRIORS = {'id': lambda data: [np.identity(len(data[0][0])) for _ in range(len(data))],
        'var': lambda data: variance(data),
        'var_1': lambda data: variance(data, axis=0),
        'cov': lambda data: covariance(data)}

print("{0}\tSTARTED  IMPORTING".format(dt.datetime.now()))
training, test = load_sorted_data()
print("{0}\tFINISHED IMPORTING".format(dt.datetime.now()))


print("{0}\tSTARTED  CALCULATING MEANS, VARs AND COV".format(dt.datetime.now()))
means = mean(training)

sigma = {}
for key in PRIORS:
    sigma[key] = PRIORS[key](training)
print("{0}\tFINISHED CALCULATING MEANS, VARs AND COV".format(dt.datetime.now()))


print("{0}\tSTARTED  CALCULATING PDFs".format(dt.datetime.now()))
pdfs = {}
for key in sigma:
    pdfs[key] = multivariates(training, sigma[key])
print("{0}\tFINISHED CALCULATING PDFs".format(dt.datetime.now()))


print("{0}\tSTARTED  PREDICTING".format(dt.datetime.now()))
preds = {}
for key in pdfs:
    print("{0}\tSTARTED WITH {1}".format(dt.datetime.now(), key))
    preds[key] = [tell_all_numbers(pdfs[key], nums) for nums in test]
print("{0}\tFINISHED PREDICTING".format(dt.datetime.now()))


training_labels = flatten_lists([[i]*len(preds['id'][i]) for i in range(10)])

confusion_matrix = {}

for key in preds:
    confusion_matrix[key] = conf_mat(flatten_lists(preds[key]), training_labels)


for key in confusion_matrix:
    file = open("{0}-confusion-matrix.txt".format(key), "w")
    norm_conf_matrix = normalize(confusion_matrix[key])
    file.write(np.array2string(norm_conf_matrix, max_line_width=np.inf))
    file.close()
