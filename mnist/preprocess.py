import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np

def load_mnist(dataset="training", digits=np.arange(10), path="data"):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    assert dataset=="training" or dataset=="testing", "dataset must be 'testing' or 'training'"

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')


    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images.astype(np.float32), labels.flatten()

def generate_target_matrix(array):
    """Generate the target matrix with probabilities based on the label array:
    if array is [3,5], then output is
    [0 0 0 1 0 0 0 0 0 0;
     0 0 0 0 0 1 0 0 0 0]
    """

    target = np.zeros((array.shape[0], 10))
    target[np.arange(0, array.shape[0]), array] = 1
    return target.astype(np.float32)

if __name__ == "__main__":
    images, labels = load_mnist()
    print(labels[0:3].flatten().shape)
    print(generate_target_matrix(labels[0:3]))
