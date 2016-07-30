from time import time
from datetime import timedelta
import os

import numpy as np

import preprocess
import network
import intraprocess
import postprocess
import stopwatch as sw
from settings import NetworkSettings

trainsettings = {
    "n_epochs": 300,
}

networksettings = NetworkSettings(
    networktype="FullyConnected",
    layersizes=(10,),
    hlnonlinearity="sigmoid",
    olnonlinearity="softmax",
    folder="networks",
    filename="test"
)

nn = network.Fullyconnected_nn(networksettings)
images_train, labels_train = preprocess.load_mnist()
images_val, labels_val = preprocess.load_mnist(dataset='testing')

targetoutput = preprocess.generate_target_matrix(labels_train)
# print(targetoutput)
t = sw.Stopwatch(start=True)
besterror = 1
besterrorepoch = 0
firstconvergederror = None
firstconvergedepoch = None
convergence = intraprocess.Convergence()

for i in range(1, trainsettings["n_epochs"]+1):
    output, loss = nn.train_fn(images_train, targetoutput)
    validationoutput = nn.eval_fn(images_val)
    results = postprocess.analyze_results(images_val, validationoutput, labels_val)
    converged = convergence.add_point(results["fraction_incorrect"])
    print("epoch {}: {} correct; done in {}. Converged: {}".format(i, results["fraction_correct"], t.get_remaining_time_formatted(i, trainsettings["n_epochs"]), converged))

    if results["fraction_incorrect"] < besterror:
        besterror = results["fraction_incorrect"]
        besterrorepoch = i
    if converged and firstconvergederror is None:
        firstconvergederror = results["fraction_incorrect"]
        firstconvergedepoch = i


    # Since initialization of the theano functions is done the first time this loop runs, the time prediction is completely off. reset the stopwatch
    if i == 1:
        t.reset(start=True)


validationoutput = nn.eval_fn(images_val)
results = postprocess.analyze_results(images_val, validationoutput, labels_val)
ind = np.arange(10) #results["incorrectly_classified"][0:10]
print("final fraction correct: {}".format(results["fraction_correct"]))

if firstconvergedepoch is None:
    firstconvergederror = results["fraction_incorrect"]
    firstconvergedepoch = trainsettings["n_epochs"]

print("convergence at {}, error: {}".format(firstconvergedepoch, firstconvergederror))
print("min erorr at {}, error: {}".format(besterrorepoch, besterror))

# print("Now saving network")
# nn.save_network()
# print("Done")

# postprocess.plot_multiple_digits(images_val[ind,:,:], results["prediction"][ind])
postprocess.plot_images_with_probabilities(images_val[ind,:,:], validationoutput[ind,:])
