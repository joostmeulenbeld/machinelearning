from time import time
from datetime import timedelta
import os

import numpy as np

import preprocess
import network
import postprocess
import stopwatch as sw

trainsettings = {
    "n_epochs": 1000,
}

networksettings = {
    "layersizes": (100, 100),
    "hlnonlinearity": "sigmoid",
    "olnonlinearity": "softmax"
}

nn = network.Fullyconnected_nn(networksettings)
images_train, labels_train = preprocess.load_mnist()
images_val, labels_val = preprocess.load_mnist(dataset='testing')

targetoutput = preprocess.generate_target_matrix(labels_train)
# print(targetoutput)
t = sw.Stopwatch(start=True)

for i in range(trainsettings["n_epochs"]):
    output, loss = nn.train_fn(images_train, targetoutput)
    validationoutput = nn.eval_fn(images_val)
    results = postprocess.analyze_results(images_val, validationoutput, labels_val)

    print("epoch {}: {} correct; done in {}".format(i+1, results["fraction"], t.get_remaining_time_formatted(i+1, trainsettings["n_epochs"])))

    # Since initialization of the theano functions is done the first time this loop runs, the time prediction is completely off. reset the stopwatch
    if i == 0:
        t.reset(start=True)


validationoutput = nn.eval_fn(images_val)
results = postprocess.analyze_results(images_val, validationoutput, labels_val)
ind = np.arange(10) #results["incorrectly_classified"][0:10]
print("final fraction correct: {}".format(results["fraction"]))
# postprocess.plot_multiple_digits(images_val[ind,:,:], results["prediction"][ind])
postprocess.plot_images_with_probabilities(images_val[ind,:,:], validationoutput[ind,:])
