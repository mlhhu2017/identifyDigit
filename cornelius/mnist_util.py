# coding: utf-8
from mnist import MNIST
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint

def get_np_array(path='data'):
    """
    Get images and install converter:
        1. install MNIST from the command line with 'pip install python-mnist'
        2. download the data from http://yann.lecun.com/exdb/mnist/
        3. extract the .gz files and rename '.' to '-' in the file names

    converts mnist images to ndarrays

    inputs:
        path            optional, path to the mnist files (default=='data')

    outputs:
        train           2d-array with shape (60000,784), training images
        train_labels    1d-array with shape (60000,),    training labels
        test            2d-array with shape (10000,784), test images
        test_labels     1d-array with shape (10000,),    test labels

    """
    mndata = MNIST(path)
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()
    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)

def show_a_num(num):
    """
    Plots a single number

    inputs:
        num    takes 1d-array with shape (784,) containing a single image

    outputs:
        img    matplotlib image

    """
    pixels = num.reshape((28,28))
    img = plt.imshow(pixels, cmap='gray')
    plt.axis("off")
    return img

def show_nums(data, nrow=None, xsize=15, ysize=15):
    """
    Plots multiple numbers in a "grid"

    inputs:
        data     takes 2d-array with shape (n,784) containing images
        nrow     optional, number of rows in the output image (default == ceil(sqrt(n)))
        xsize    optional, specifies output image length in inches (default == 15)
        ysize    optional, specifies output image height in inches (default == 15)

    outputs:
        img      matplotlib image

    """
    n = len(data)
    # check if at least one image
    if n < 1:
        raise ValueError("No image given!")

    # if only 1 image print it
    if len(data.shape) == 1:
        return show_a_num(data)

    # number of rows specified?
    if nrow == None:
        # calculate default
        ncol = math.ceil(math.sqrt(n))
        nrow = math.ceil(n/ncol)
    else:
        # calculate number of columns
        ncol = math.ceil(n/nrow)

    # check if enough images
    missing = nrow*ncol - n
    if missing != 0:
        # fill up with black images
        zeros = np.zeros(missing*784)
        zeros = zeros.reshape(missing,784)
        data = np.vstack((data,zeros))

    # reshape the data to the desired output
    data = data.reshape((-1,28,28))
    data = data.reshape((nrow,-1,28,28)).swapaxes(1,2)
    data = data.reshape((nrow*28,-1))

    plt.figure(figsize=(xsize,ysize))
    img = plt.imshow(data, cmap='gray')
    plt.axis("off")
    return img

def get_one_num(data, labels, num):
    """
    Creates 2d-array containing only images of a single number

    inputs:
        data      takes 2d-array with shape (n,784) containing the images
        labels    takes 1d-array with shape (n,)    containing the labels
        num       the number you want to filter

    outputs:
        arr       2d-array only containing a images of num

    """
    return np.array([val for idx, val in enumerate(data) if labels[idx] == num])

def get_all_nums(data, labels):
    """
    Creates a 1d-array containing 2d-arrays of images for every number
    ex. arr[0] = 2d-array containing all images of number 0

    inputs:
        data      takes 2d-array with shape (n,784) containing the images
        labels    takes 1d-array with shape (n,)    containing the labels

    outputs:
        arr       1d-array containing 2d-arrays for every number

    """
    return np.array([get_one_num(data, labels, i) for i in range(10)])

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    inputs:
        cm           confusion matrix
        classes      name of classes
        normalize    optional, normalize matrix to show percentages (default == False)
        title        title of the plot (default == 'Confusion matrix')
        cmap         colormap (default == blue colormap)

    outputs:
        void

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
