"""Module to handle the mnist dataset easier with a probabilistic approach."""

import itertools
import matplotlib.pyplot as plot
import numpy as np

from mnist import MNIST
from sklearn.datasets import fetch_mldata
from scipy.stats import multivariate_normal as multivar

def load_data(path='data'):
    """Function to retrieve the MNIST dataset.

    Return:
        (training_img, training_target, test_img, test_target)"""

    mndata = MNIST(path)
    training_img, training_target = mndata.load_training()
    test_img, test_target = mndata.load_testing()

    return training_img, training_target, test_img, test_target

def load_sorted_data(path='data'):
    """Function to retrieve the MNIST dataset sorted."""
    t_img, t_target, test_img, test_target = load_data(path)

    training = [filter_number(t_img, t_target, i) for i in range(10)]
    test = [filter_number(test_img, test_target, i) for i in range(10)]

    return training, test

def mnist():
    """Function to retrieve the MNIST dataset.

    Return:
        (data, target)
            data: An array with the length of 70000 where the entries are also
            arrays with the size of 784.
            target: The corresponding value/solution to the piece of data.
    """
    data = fetch_mldata('MNIST original', data_home='.')
    return data.data, data.target

def mnist_splitted():
    """Splitts the mnist dataset into a training and test set."""
    data, target = mnist()
    filtered = [filter_number(data, target, i) for i in range(10)]

    training = [filtered[i][:6000] for i in range(10)]
    test = [filtered[i][6000:] for i in range(10)]

    return training, test

def filter_number(data, target, number):
    """Filters the given data for the wanted number."""
    return [x for x, y in zip(data, target) if y == number]

def plot_number(number_vec,figsize=(10,10) ):
    """Reshapes a vector with shape (784, ) to (28, 28) and draws the image."""
    plot.figure(figsize=figsize)
    plot.imshow(number_vec.reshape(28, 28), cmap='Greys')
    plot.show()

def plot_all_numbers(numbers, elements_per_line=10, scale=True, plot_title=""):
    """Takes a list of arrays and draws them.

    Args:
        numbers: List of arrays with the shape (784, )
        lines (optional): The amount of lines you want to have drawn.
    """
    lines = chunk(numbers, elements_per_line)

    needed_amount = elements_per_line - len(lines[-1])
    filling_space = [np.zeros(28*28) for _ in range(needed_amount)]
    lines[-1].extend(filling_space)

    reshape_lam = lambda x: np.concatenate([y.reshape(28, 28).T for y in x]).T
    tmp = [reshape_lam(x) for x in lines]

    if scale is False:
        plot.figure()
    else:
        plot.figure(figsize=(15, 15*len(lines)))

    plot.title(plot_title)
    plot.imshow(np.concatenate(tmp, axis=0), cmap='Greys')
    plot.show()

def chunk(some_list, size):
    """Returns a list of lists which have the size of n. The last list may have
    less elements.

    Args:
        some_list: A list.
        size: The size of the chunks you want to have.

    Return:
        A list of list, where each list has the given size, except the last
        which can have some elements less."""

    chunks = []
    limit = int(np.ceil(len(some_list)/size))

    for i in range(limit):
        if (i+1)*size > len(some_list):
            chunks.append(some_list[i*size : len(some_list)])
        else:
            chunks.append(some_list[i*size : (i+1)*size])

    return chunks

def mean(datasets):
    """Calculates the means of a list of lists which contain here arrays."""
    return [np.mean(x, axis=0) for x in datasets]

def variance(datasets, axis=None):
    """Calculates the variance of a list of lists which contain here arrays."""
    return [np.var(x, axis=axis) for x in datasets]

def covariance(datasets):
    """Calculates the covariance of a list of lists which contain here arrays."""
    return [np.cov(np.array(datasets[i]).T) for i in range(len(datasets))]

def multivariates(datasets, covar, means=None):
    """Does stuff"""

    if means is None:
        means = mean(datasets)

    return [multivar(mean=means[i], cov=covar[i], allow_singular=True).logpdf\
                for i in range(len(datasets))]

def tell_number(pdfs, number):
    """Does more stuff."""
    pred = [pdf(number) for pdf in pdfs]
    return pred.index(max(pred))

def tell_all_numbers(pdfs, numbers):
    """Does more than tell_number."""
    return [tell_number(pdfs, x) for x in numbers]

# https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
def flatten_lists(lists):
    """Flattens a list of lists"""
    return [item for sublist in lists for item in sublist]

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """    

    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.xlabel('True label')
    plot.ylabel('Predicted label')

def normalize(matrix):
    return matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]


