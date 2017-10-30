
# coding: utf-8

# In[1]:


from mnist import MNIST
import numpy as np

# 1. install MNIST from the command line with 'pip install python-mnist'
# 2. download the data from http://yann.lecun.com/exdb/mnist/
# 3. extract the .gz files and rename the files the '.' to '-'
# Takes a path to data as argument. Looks for the data in the folder 'data/' if called without an argument.
# Returns np arrays of trainings_images, trainings_labels, test_images, and test_labels in that order.
def get_np_array(path='data'):
    mndata = MNIST(path)
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_training()
    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels)

train, train_labels, test, test_labels = get_np()


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

# shows image of a single number
# needs a row with length 784 as argument
def show_a_num(num):
    pixels = num.reshape((28,28))
    #plt.title("Label: {label}".format(label=label))
    plt.imshow(pixels, cmap='gray')


# In[3]:


# shows images of multiple numbers in a "grid"
# need a 2d np array with rows of length 784 as argument
def show_nums(data):
    factors = lambda n: set([i for i in range(1, int(n**0.5) + 1) if n % i == 0])
    fac = factors(len(data))
    nrow = max(list(fac))
    data = data.reshape((-1,28,28))
    data = data.reshape((nrow,-1,28,28)).swapaxes(1,2)
    data = data.reshape((nrow*28,-1))
    plt.figure(figsize=(15,15))
    plt.imshow(data, cmap='gray')


# In[4]:


from random import randint

rnd = randint(0,len(train)-1)
print("showing picture number: {rnd}".format(rnd=rnd))
show_a_num(train[rnd])


# In[63]:


test = train[:100]
show_nums(test)
