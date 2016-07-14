import theano
import theano.tensor as T
import lasagne
import numpy as np

import postprocess
import preprocess



class fullyconnected_nn(object):
    def __init__(self):
        self._input = T.tensor3('input')
        self._target = T.matrix('target')
        
        self._network = self.network_nn(input_var=self._input)

        params = lasagne.layers.get_all_params(self._network['output'])
        output = lasagne.layers.get_output(self._network['output'])
    
        loss = lasagne.objectives.categorical_crossentropy(output, self._target)
        loss = lasagne.objectives.aggregate(loss, mode='mean')
        
        updates = lasagne.updates.adadelta(loss, params, learning_rate=1, rho=0.9, epsilon=1e-06)
        updates = lasagne.updates.apply_nesterov_momentum(updates,params,momentum=0.9)

        print("creating evaluation function")
        self._eval_fn = theano.function([self._input], output)
        print("creating validation function")
        self._val_fn = theano.function([self._input, self._target], [output, loss])
        print("creating train function")
        self._train_fn = theano.function([self._input, self._target], [output, loss], updates=updates)
            
    def validation_error(self, validation_data, targetlabels):
        output = self._eval_fn(validation_data)
        prediction = np.argmax(output, axis=1)

        fraction_correct = np.count_nonzero(prediction==targetlabels)/targetlabels.shape[0]
        return fraction_correct
        
    def network_nn(self, input_var=None, image_size=(28,28)):
        network = {}
        network['input'] = lasagne.layers.InputLayer((None, image_size[0], image_size[1]), input_var=input_var)
        network['fcl1'] = lasagne.layers.DenseLayer(network['input'], num_units=50, nonlinearity=lasagne.nonlinearities.sigmoid)
        network['output'] = lasagne.layers.DenseLayer(network['fcl1'], num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
        return network

# class cnn(object):
#     def __init__():
#         self._network = self.get_network()
#         self.trainfunction = 
# 
#     def network_cnn(self, image_size=(28,28), filter_size=(3,3)):
#         network = {}
#         network['input'] = lasagne.layers.InputLayer(shape=(None, 1, image_size[0], image_size[1]), filter_size=filter_size)
#         network['conv1'] = lasagne.layers.

if __name__ == "__main__":
    nn = fullyconnected_nn()
    images_train, labels_train = preprocess.load_mnist()
    images_val, labels_val = preprocess.load_mnist(dataset='testing')

    targetoutput = preprocess.generate_target_matrix(labels_train)
    # print(targetoutput)
    for i in range(100):
        output, loss = nn._train_fn(images_train, targetoutput)
        print(nn.validation_error(images_val, labels_val))
        
    # print("label: {}".format(labels[20000]))
    # print("output:")
    # print(nn._eval_fn(np.expand_dims(images[20000,:,:], 0)))
    # postprocess.plot_digit(images[20000,:,:])
