# coding: utf-8
from mnist import MNIST
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint

# 1. install MNIST from the command line with 'pip install python-mnist'
# 2. download the data from http://yann.lecun.com/exdb/mnist/
# 3. extract the .gz files and rename '.' to '-' in the file names
#
# get_np_array converts mnist images to ndarrays
#
# inputs:
#   path            optional, path to the mnist files (default=='data')
#
# outputs:
#   train           2d-array with shape (60000,784), training images
#   train_labels    1d-array with shape (60000,),    training labels
#   test            2d-array with shape (10000,784), test images
#   test_labels     1d-array with shape (10000,),    test labels
#
def get_np_array(path='data'):
    mndata = MNIST(path)
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()
    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)

# show_a_num plots a single picture
#
# inputs:
#   num    takes 1d-array with shape (784,) containing a single image
#
# outputs:
#   img    matplotlib image
#
def show_a_num(num):
    pixels = num.reshape((28,28))
    img = plt.imshow(pixels, cmap='gray')
    return img

# show_nums plots multiple pictures in a "grid"
#
# inputs:
#   data      takes 2d-array with shape (n,784) containing images
#   square    optional, boolean, if 'True' forces a square output image (default==False)
#   xsize     optional, specifies output image length in inches (default==15)
#   ysize     optional, specifies output image height in inches (default==15)
#
# outputs:
#   img       matplotlib image
#
def show_nums(data,square=False,xsize=15,ysize=15):
    if len(data.shape) == 1:
        return show_a_num(data)
    n = len(data)
    if square:
        nrow = math.ceil(math.sqrt(n))
        missing = nrow**2 - n
        zeros = np.zeros(missing*784)
        zeros = zeros.reshape(missing,784)
        data = np.vstack((data,zeros))
    else:
        factors = lambda n: set([i for i in range(1, int(n**0.5) + 1) if n % i == 0])
        fac = factors(n)
        nrow = max(list(fac))
    data = data.reshape((-1,28,28))
    data = data.reshape((nrow,-1,28,28)).swapaxes(1,2)
    data = data.reshape((nrow*28,-1))
    plt.figure(figsize=(xsize,ysize))
    img = plt.imshow(data, cmap='gray')
    return img

# get_one_num: creates 2d-array containing only images of a single number
#
# inputs:
#   data      takes 2d-array with shape (n,784) containing the images
#   labels    takes 1d-array with shape (n,)    containing the labels
#   num       the number you want to filter
#
# outputs:
#   arr       2d-array only containing a sinlge number
#
def get_one_num(data,labels,num):
    return np.array([data[i] for i in range(len(data)) if labels[i] == num])

# get_all_nums: creates a 1d-array containing 2d-arrays for every number
#              ex. arr[0] = 2d-array containing all images of the number 0
#
# inputs:
#   data      takes 2d-array with shape (n,784) containing the images
#   labels    takes 1d-array with shape (n,)    containing the labels
#
# outputs:
#   arr       1d-array containing 2d-arrays for every number
#
def get_all_nums(data,labels):
    return np.array([get_one_num(data,labels,i) for i in range(10)])
