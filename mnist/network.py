import theano
import theano.tensor as T
import lasagne
import numpy as np

import postprocess
import preprocess

nonlinearities = {
    "sigmoid": lasagne.nonlinearities.sigmoid,
    "softmax": lasagne.nonlinearities.softmax
}

class Fullyconnected_nn(object):
    """class of a fully connected layer for mnist recognition"""

    def __init__(self, networksettings):
        self.networksettings = networksettings
        self._input = T.tensor3('input')
        self._target = T.matrix('target')

        hlnonlinearity = nonlinearities[self.networksettings["hlnonlinearity"]]
        olnonlinearity = nonlinearities[self.networksettings["olnonlinearity"]]
        self._network = self.network_nn(layersizes=self.networksettings["layersizes"], hlnonlinearity=hlnonlinearity, olnonlinearity=olnonlinearity)

        self._params = lasagne.layers.get_all_params(self._network[-1])
        self._output = lasagne.layers.get_output(self._network[-1])

        self._loss = lasagne.objectives.categorical_crossentropy(self._output, self._target)
        self._loss = lasagne.objectives.aggregate(self._loss, mode='mean')

        self._updates = lasagne.updates.adadelta(self._loss, self._params, learning_rate=1, rho=0.9, epsilon=1e-06)
        self._updates = lasagne.updates.apply_nesterov_momentum(self._updates, self._params, momentum=0.9)

        self.__eval_fn = None
        self.__val_fn = None
        self.__train_fn = None


    @property
    def eval_fn(self):
        """getter of evaluation function. Only generate the function when needed, saves time during initial loading"""
        if self.__eval_fn is None:
            print("creating evaluation function")
            self.__eval_fn = theano.function([self._input], self._output)
        return self.__eval_fn

    @property
    def val_fn(self):
        """getter of validation function. Only generate the function when needed, saves time during initial loading"""
        if self.__val_fn is None:
            print("creating validation function")
            self.__val_fn = theano.function([self._input, self._target], [self._output, self._loss])
        return self.__val_fn

    @property
    def train_fn(self):
        """getter of training function. Only generate the function when needed, saves time during initial loading"""
        if self.__train_fn is None:
            print("creating training function")
            self.__train_fn = theano.function([self._input, self._target], [self._output, self._loss], updates=self._updates)
        return self.__train_fn

    def network_nn(self, layersizes=(50,), hlnonlinearity=nonlinearities["sigmoid"], olnonlinearity=nonlinearities["softmax"], image_size=(28,28)):
        """Create the fully connected network
        INPUT:
            layersizes: tuple containing the amount of neurons per hidden layer. amount of layers is taken from len(layersizes)
            hlnonlinearity: nonlinearity of hidden layer
            olnonlinearity: nonlinearity of output layer
            images_size=(28,28): tuple containing the input dimension
        """
        network = []
        network.append(lasagne.layers.InputLayer((None, image_size[0], image_size[1]), input_var=self._input))
        for layersize in layersizes:
            network.append(lasagne.layers.DenseLayer(network[-1], num_units=layersize, nonlinearity=hlnonlinearity))
        network.append(lasagne.layers.DenseLayer(network[-1], num_units=10, nonlinearity=olnonlinearity))
        return network



if __name__ == "__main__":
    nn = Fullyconnected_nn()
    images_train, labels_train = preprocess.load_mnist()
    images_val, labels_val = preprocess.load_mnist(dataset='testing')

    targetoutput = preprocess.generate_target_matrix(labels_train)
    # print(targetoutput)
    for i in range(10):
        output, loss = nn.train_fn(images_train, targetoutput)
        print("epoch {}".format(i+1))
