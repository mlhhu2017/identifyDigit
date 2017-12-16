import numpy as np
import matplotlib.pyplot as plt
import itertools
import logging as l

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mnist_util import *

training, test = load_sorted_data('data_notMNIST') 
means = mean(training)

id = [np.identity(28*28) for _ in range(len(training))]
variances = variance(training)
variances1 = variance(training, axis=0)
covariances = covariance(training)

plot_all_numbers(means, elements_per_line=5, plot_title="Means of the training dataset")
plot_all_numbers(variances1, elements_per_line=5, plot_title="Variances of the training dataset")

pdfs_id = multivariates(training, id)
pdfs_var = multivariates(training, variances)
pdfs_var_1 = multivariates(training, variances1)
pdfs_cov = multivariates(training, covariances)

tmp = flatten_lists([test[i][:20] for i in range(10)])
tmp = [np.array(x) for x in tmp]
plot_all_numbers(tmp, elements_per_line=10, plot_title="First 20 of each number from the test dataset")
#for i in range(10):
#    plot_all_numbers(test[i][:20], elements_per_line=10, plot_title="Test dataset: Number {0}".format(i))

for i in range(10):
    guess_identity = [tell_number(pdfs_id, num) for num in test[i][:20]]
    guess_variance = [tell_number(pdfs_var, num) for num in test[i][:20]]
    guess_variance_1 = [tell_number(pdfs_var_1, num) for num in test[i][:20]]
    guess_covariance = [tell_number(pdfs_cov, num) for num in test[i][:20]]
    
    print("Right guess: {0}".format(i))
    print("id:\t{0}\tERRORS: {1}".format(guess_identity, len([x for x in guess_identity if x != i])))
    print("var:\t{0}\tERRORS: {1}".format(guess_variance, len([x for x in guess_variance if x != i])))
    print("var1:\t{0}\tERRORS: {1}".format(guess_variance_1, len([x for x in guess_variance_1 if x != i])))
    print("cov:\t{0}\tERRORS: {1}".format(guess_covariance, len([x for x in guess_covariance if x != i])))
    print("")

guess_identity = []
guess_variance = []
guess_variance_1 = []
guess_covariance = []
for i in range(10):
    guess_identity.append([tell_number(pdfs_id, num) for num in test[i]])
    guess_variance.append([tell_number(pdfs_var, num) for num in test[i]])
    guess_variance_1.append([tell_number(pdfs_var_1, num) for num in test[i]])
    guess_covariance.append([tell_number(pdfs_cov, num) for num in test[i]])
    
    print("Number: {0}\tAMOUNT: {1}".format(i, len(test[i])))
    print("identity:\tERRORS: {0}".format(len([x for x in guess_identity[i] if x != i])))
    print("variance:\tERRORS: {0}".format(len([x for x in guess_variance[i] if x != i])))
    print("variance_1:\tERRORS: {0}".format(len([x for x in guess_variance_1[i] if x != i])))
    print("covariance:\tERRORS: {0}".format(len([x for x in guess_covariance[i] if x != i])))
    print("")

training_labels = flatten_lists([[i]*len(guess_identity[i]) for i in range(10)])
guess_identity_flat = flatten_lists(guess_identity)
guess_variance_flat = flatten_lists(guess_variance)
guess_variance_1_flat = flatten_lists(guess_variance_1)
guess_covariance_flat = flatten_lists(guess_covariance)

confusion_matrix_identity = confusion_matrix(guess_identity_flat, training_labels)
confusion_matrix_variance = confusion_matrix(guess_variance_flat, training_labels)
confusion_matrix_variance_1 = confusion_matrix(guess_variance_1_flat, training_labels)
confusion_matrix_covariance = confusion_matrix(guess_covariance_flat, training_labels)

class_names = [str(i) for i in range(10)]

plt.figure()
plot_confusion_matrix(confusion_matrix_identity, classes=class_names,normalize=True, title='Identity Confusion Matrix')

plt.figure()
plot_confusion_matrix(confusion_matrix_variance, classes=class_names,normalize=True, title='Variance Confusion Matrix')

plt.figure()
plot_confusion_matrix(confusion_matrix_variance_1, classes=class_names,normalize=True, title='Variance_1 Confusion Matrix')

plt.figure()
plot_confusion_matrix(confusion_matrix_covariance, classes=class_names,normalize=True, title='Covariance Confusion Matrix')

plt.show()
guess_variance_1_flat = flatten_lists(guess_variance_1)
guess_covariance_flat = flatten_lists(guess_covariance)
