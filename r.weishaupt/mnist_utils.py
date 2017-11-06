# Import required packages
import numpy as np;
import idx2numpy as idx;
import matplotlib.pyplot as plt;

def loadset(data, labels):
    """ Loads data and labels from given paths.

    Arguments:
        data [string] -- Path to data file
        labels [string] -- Path to labels file

    Return:
        [2-tuple:np.array] -- Returns tuple of numpy-arrays,
        first one is the data set, second one is the label set
    """
    return tuple(map(idx.convert_from_file, [data, labels]));

def showimg(img, plotsize=[3,3]):
    """ Plots a single image as grayscaled pixel map through matplotlib.pyplot.

    Arguments:
        img [np.array] -- Matrix of integers to be interpreted as
        pixels of an image
        plotsize [list(integer)] -- Sets the size of the plot for the image.

    Return:
        [void]
    """
    # Create new canvas and plot image without axes
    fig = plt.figure(figsize=plotsize);
    plt.imshow(img, cmap="gray");
    plt.axis("off");

def showimgset(imgs, y = None, dim = 28, plotsize=[10,5]):
    """ Plots a set of n images positioned in a y*x-grid through matplotlib.pyplot.

    Arguments:
        imgs [np.array] -- List of n images
        y [int] -- Number of rows, defaults to ceil(sqrt(n))
        dim [int] -- The dimension of a single image is dim x dim, defaults to 28
        plotsize [list(integer)] -- Sets the size of the plot for the set of images.

    Return:
        [void]
    """
    # Number of images recieved
    k = len(imgs);

    # At least one image is required
    if k < 1:
        raise ValueError("No image given!");

    # Did we recieve a value for y?
    if y == None:
        # Calculate default value
        y = int(np.ceil(np.sqrt(k)));
    # Calculate x value based on y
    x = int(np.ceil(k / y));

    # Are there enough images given?
    if k != x*y:
        # Fill up with black images
        imgs = np.concatenate((imgs, np.zeros((x*y-k, dim, dim))));

    # Reshape to (y*28)x(x*28)-array
    imgmap = np.vstack([np.hstack(imgs[j*x:(j+1)*x]) for j in range(y)]);

    # Create new canvas, plot image map
    fig = plt.figure(figsize=plotsize);
    plt.imshow(imgmap, cmap="gray");
    plt.axis("off");

def getdigit(digit, data, labels):
    """ Returns the first image representing a digit of type num.
    Data and labels must be in the same order!

    Arguments:
        digit [integer] -- Type of digit to be returned
        data [np.array] -- List of all digits
        labels [np.array] -- List of all labels

    Return:
        [np.array] -- Returns a single image as np.array
    """
    # Search for first occurence of digit in labels
    i = 0;
    while True:
        # Did we find the right one?
        if labels[i] == digit:
            break;
        # Check next one
        i += 1;
    # Return corresponding image
    return data[i];

def getalldigits(digit, data, labels, n=None):
    """ Returns list of n images all representing the searched digit.

    Arguments:
        digit [integer] -- Type of digit to be returned
        data [np.array] -- List of all digits
        labels [np.array] -- List of all labels
        n [integer] -- Amount of digits to be returned, defaults to all

    Return:
        [np.array] - List of images representing given digit
    """
    # Return all entries in data where corresponding labels entry is digit.#
    return data[np.where(labels == digit)[0][:n]]

def showconfmatrix(matrix, labels, plotsize=[10,10]):
    """ Plots a confusion matrix based on the given input matrix.

    Arguments:
        matrix [np.array] -- A nxn array of frequency counts
        labels [np.array] -- An array of n labels
        plotsize [list(integer)] -- Sets the size of the plot for the image.

    Return:
        [void]
    """
    # Save matrix shape
    x, y = matrix.shape;
    # Create new canvas
    fig = plt.figure(figsize=plotsize);
    # Add suplot to canvas
    ax = fig.add_subplot(111);
    # Display matrix
    img = ax.imshow(matrix, cmap=plt.cm.Pastel2);

    # Add labels to fields
    for i in range(x):
        for j in range(y):
            ax.annotate(str(matrix[i][j]), xy=(j,i),
                horizontalalignment="center",
                verticalalignment="center");

    # Add color bar to the right
    cb = fig.colorbar(img);
    # Add labels to the axes
    plt.xticks(range(x), labels[:x]);
    plt.yticks(range(y), labels[:y]);
