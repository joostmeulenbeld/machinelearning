import numpy as np
import matplotlib.pyplot as plt

import preprocess


def plot_digit(digit):
    """ show the matrix of a single digit in a plot"""
    plt.imshow(digit, cmap=plt.get_cmap("gray"), interpolation="none")
    plt.show()

def plot_multiple_digit(digits):
    """ show the matrix of multiple digits in a matrix of plots. This function makes a grid
    as square as possible and fills it with the digits. Retention of order is not guaranteed"""
    height = int(np.ceil(np.sqrt(digits.shape[0])))
    width = int(np.ceil(digits.shape[0]/height))
    fig = plt.figure()

    for i in range(digits.shape[0]):
        plt.subplot(height, width, i)
        plt.imshow(digits[i,:,:], cmap=plt.get_cmap("gray_r"), interpolation="none")
        plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    path = 'data'

    images, labels = preprocess.load_mnist(path=path)
    postprocess.plot_multiple_digit(images[0:200,:,:])
