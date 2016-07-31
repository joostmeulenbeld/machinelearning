import os
try:
    import cPickle as pickle
except:
    import pickle

import theano
import theano.tensor as T
import lasagne
import numpy as np

import postprocess
import preprocess

def load_network(filename):
    """ Load a network from the pickled object. filename is automatically appended .pkl if needed
    """
    if ".pkl" not in filename:
        filename += ".pkl"

    f = open(filename, 'rb')
    struct = pickle.load(f)
    if struct["networksettings"].networktype == "FullyConnected":
        return Fullyconnected_nn(struct["networksettings"], struct["parameters"])

class Fullyconnected_nn(object):
    """class of a fully connected layer for mnist recognition"""

    def __init__(self, networksettings, parameters=None):

        assert networksettings.networktype == "FullyConnected", "This class is a fully connected network, networksettings specify {}".format(networksettings["networktype"])
        self.networksettings = networksettings

        self._input = T.tensor3('input')
        self._target = T.matrix('target')

        self._network = self.network_nn()

        self._params = lasagne.layers.get_all_params(self._network[-1])
        self._output = lasagne.layers.get_output(self._network[-1])

        self._loss = lasagne.objectives.categorical_crossentropy(self._output, self._target)
        self._loss = lasagne.objectives.aggregate(self._loss, mode='mean')

        self._updates = lasagne.updates.adadelta(self._loss, self._params, learning_rate=1, rho=0.9, epsilon=1e-06)
        self._updates = lasagne.updates.apply_nesterov_momentum(self._updates, self._params, momentum=0.9)

        if parameters is not None:
            self.set_parameters(parameters)

        self.__eval_fn = None
        self.__val_fn = None
        self.__train_fn = None

    def set_parameters(self, parameters):
        """set layer weights and biases etc)"""
        lasagne.layers.set_all_param_values(self._network, parameters)

    def get_parameters(self):
        """get layer weights and biases etc)"""
        return lasagne.layers.get_all_param_values(self._network)

    @property
    def eval_fn(self):
        """getter of evaluation function. Only generate the function when needed, saves time during initial loading"""
        if self.__eval_fn is None:
            # print("creating evaluation function")
            self.__eval_fn = theano.function([self._input], self._output)
        return self.__eval_fn

    @property
    def val_fn(self):
        """getter of validation function. Only generate the function when needed, saves time during initial loading"""
        if self.__val_fn is None:
            # print("creating validation function")
            self.__val_fn = theano.function([self._input, self._target], [self._output, self._loss])
        return self.__val_fn

    @property
    def train_fn(self):
        """getter of training function. Only generate the function when needed, saves time during initial loading"""
        if self.__train_fn is None:
            # print("creating training function")
            self.__train_fn = theano.function([self._input, self._target], [self._output, self._loss], updates=self._updates)
        return self.__train_fn

    def network_nn(self, image_size=(28,28)):
        """Create the fully connected network, settings are taken from self.networksettings
        INPUT:
            images_size=(28,28): tuple containing the input dimension
        """
        network = []

        network.append(lasagne.layers.InputLayer((None, image_size[0], image_size[1]), input_var=self._input))

        for layersize in self.networksettings.layersizes:
            network.append(lasagne.layers.DenseLayer(network[-1], num_units=layersize, nonlinearity=self.networksettings.get_hlnonlinearity()))

        network.append(lasagne.layers.DenseLayer(network[-1], num_units=10, nonlinearity=self.networksettings.get_olnonlinearity()))

        return network

    def save_network(self, appendfilename=""):
        """ Save the networksettings and weights/biases to file.
        INPUT:
            appendfilename: append the filename specified in networksettings with this string
            note that the filename is automatically appended with .pkl
        """
        if not os.path.exists(self.networksettings.folder):
            print("Creating directory {}".format(self.networksettings.folder))
            os.makedirs(self.networksettings.folder)

        struct = {
            "networksettings": self.networksettings,
            "parameters": self.get_parameters()
        }
        filename = self.networksettings.get_savename() + appendfilename
        if '.pkl' not in filename:
            filename += '.pkl'

        f = open(filename, 'wb')
        pickle.dump(struct, f)
        print("Saved the network to {}".format(filename))


if __name__ == "__main__":
    images_train, labels_train = preprocess.load_mnist()
    images_val, labels_val = preprocess.load_mnist(dataset='testing')

    nn = load_network(os.path.join("networks", "test"))

    # targetoutput = preprocess.generate_target_matrix(labels_train)
    # # print(targetoutput)
    # for i in range(10):
    #     output, loss = nn.train_fn(images_train, targetoutput)
    #     print("epoch {}".format(i+1))
