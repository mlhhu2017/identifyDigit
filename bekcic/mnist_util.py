"""Module to handle the mnist dataset easier with a probabilistic approach."""

import numpy as np

from matplotlib.pyplot import imshow, show, title
from sklearn.datasets import fetch_mldata
from scipy.stats import multivariate_normal as multivar

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

def plot_number(number_vec):
    """Reshapes a vector with shape (784, ) to (28, 28) and draws the image."""
    imshow(number_vec.reshape(28, 28), cmap='Greys')
    show()

def plot_all_numbers(numbers, elements_per_line=10, plot_title=""):
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

    title(plot_title)
    imshow(np.concatenate(tmp, axis=0), cmap='Greys')
    show()

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

def multivariates(datasets, covar):
    """Does stuff"""
    means = mean(datasets)
    return [multivar(mean=means[i], cov=covar[i], allow_singular=True).logpdf for i in range(len(datasets))]

def tell_number(pdfs, number):
    """Does more stuff."""
    pred = [pdf(number) for pdf in pdfs]
    return pred.index(max(pred))
