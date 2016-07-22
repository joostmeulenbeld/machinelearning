import numpy as np
import matplotlib.pyplot as plt

import preprocess
import network

def plot_digit(digit):
    """ show the matrix of a single digit in a plot"""
    plt.imshow(digit, cmap=plt.get_cmap("gray"), interpolation="none")
    plt.show()

def plot_multiple_digits(images, labels=None):
    """ show the matrix of multiple images in a matrix of plots. This function makes a grid
    as square as possible and fills it with the images. Retention of order is not guaranteed
    If labels is not None, every image will be shown along with the label of corresponding index"""

    if labels is not None:
        assert labels.size == images.shape[0], "Number of labels does not match number of images"

    height = int(np.ceil(np.sqrt(images.shape[0])))
    width = int(np.ceil(images.shape[0]/height))
    fig = plt.figure()

    for i in range(images.shape[0]):
        ax = plt.subplot(height, width, i)
        ax.imshow(images[i,:,:], cmap=plt.get_cmap("gray_r"), interpolation="none")
        ax.axis('off')
        if labels is not None:
            ax.set_title(str(labels[i]))

    plt.show()

def plot_images_with_probabilities(images, predictiondistribution):
    """ Show images along with their predicted distribution
    INPUT:
        images: the images
        predictiondistribution: the output distribution of the network
    """
    assert images.shape[0] == predictiondistribution.shape[0], "Number of images does not match the amount of prediction labels"
    assert images.shape[0] <= 20, "Can't show more than 20 images"

    fig, axarr = plt.subplots(images.shape[0], 2)
    for i in range(0, images.shape[0], ):
        #Plot the image itself
        ax = axarr[i,0]
        ax.imshow(images[i,:,:],
            cmap=plt.get_cmap("gray_r"),
            interpolation="none")
        ax.axis("off")

        #Plot the probability distribution
        ax = axarr[i,1]
        ax.imshow(np.expand_dims(predictiondistribution[i,:], 0), 
            cmap=plt.get_cmap("gray"),
            interpolation="none",
            vmin=0,
            vmax=1)
        ax.axes.get_xaxis().set_ticks(np.arange(10))
        ax.axes.get_yaxis().set_ticks([])

    plt.show()


def analyze_results(images, predictiondistribution, targetlabels):
    """ Analyze the results of the output of the network
    INPUT:
        images: the images
        predictionalabels: the probabilities
    OUTPUT:
        dict with the following keys:
            prediction: the labels extracted from the probabilities output by the network
            correctly_classified: indices of images that were correctly correctly_classified
            fraction: fraction of numbers that were correctly classified
    """

    results = {}
    results["prediction"] = np.argmax(predictiondistribution, axis=1)
    results["correctly_classified"] = np.nonzero(results["prediction"] == targetlabels)[0]
    results["incorrectly_classified"] = np.nonzero(results["prediction"] != targetlabels)[0]
    results["fraction"] = results["correctly_classified"].size / targetlabels.shape[0]

    return results


if __name__ == "__main__":
    path = 'data'

    images, labels = preprocess.load_mnist(path=path)
    # results = analyze_results(images, )
    plot_multiple_digits(images[0:30,:,:], labels[0:30])
