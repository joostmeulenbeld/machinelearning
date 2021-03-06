from time import time
from datetime import timedelta
import os
# os.environ["THEANO_FLAGS"] = "floatX=float32,device=cpu,dnn.enabled=False,optimizer_including=conv_meta,metaopt.verbose=1,optimizer_excluding=more_mem"

import numpy as np

import preprocess
import network
import intraprocess
import postprocess
import stopwatch as sw
from settings import NetworkSettings

no_neurons_to_test = 3

# neuronresults = np.zeros((no_neurons_to_test,1))
# bestepochnumbers = np.zeros((no_neurons_to_test,1))
# bestepochtimes = np.zeros((no_neurons_to_test,1))
#
# for neurons in range(1,no_neurons_to_test+1):



totaltime = sw.Stopwatch(start=True)

trainsettings = {
    "n_epochs": 1000000,
    "convergence_n_epochs": 300
}

networksettings = NetworkSettings(networktype="FullyConnected", layersizes=(60, 50), hlnonlinearity="sigmoid",
                                  olnonlinearity="softmax", folder="networks", filename="test")



nn = network.Fullyconnected_nn(networksettings)
images_train, labels_train = preprocess.load_mnist()
images_val, labels_val = preprocess.load_mnist(dataset='testing')

targetoutput = preprocess.generate_target_matrix(labels_train)
# print(targetoutput)
besterror = 1
besterrorepoch = 0
firstconvergederror = None
firstconvergedepoch = None
convergence = intraprocess.Convergence(trainsettings["convergence_n_epochs"])
t = sw.Stopwatch(start=True)

for i in range(1, trainsettings["n_epochs"]+1):
    output, loss = nn.train_fn(images_train, targetoutput)
    validationoutput = nn.eval_fn(images_val)
    results = postprocess.analyze_results(images_val, validationoutput, labels_val)
    if convergence.add_point(results["fraction_incorrect"]):
        besterror = results["fraction_incorrect"]
        bestparameters = nn.get_parameters()
    if convergence.converged():
        break
    print("epoch {}: {} correct; done in {}. Converged: {}, best_epoch: {}, {} epochs till converged".format(
        i,
        results["fraction_correct"],
        t.get_remaining_time_formatted(i, trainsettings["n_epochs"]),
        convergence.converged(),
        convergence.best_error_epoch,
        trainsettings["convergence_n_epochs"] - (i - convergence.best_error_epoch)))

    # Since initialization of the theano functions is done the first time this loop runs, the time prediction is completely off. reset the stopwatch
    if i == 1:
        t.reset(start=True)

# print("Setting network parameters to best values encountered at epoch {}".format(besterrorepoch))
nn.set_parameters(bestparameters)

validationoutput = nn.eval_fn(images_val)
results = postprocess.analyze_results(images_val, validationoutput, labels_val)
ind = np.arange(10) #results["incorrectly_classified"][0:10]
print("Final error: {}. epochs: {}, time elapsed: {}".format(results["fraction_incorrect"], convergence.best_error_epoch, totaltime.stop()))
# neuronresults[neurons-1] = results["fraction_incorrect"]
# bestepochtimes[neurons-1] = totaltime.get_time()
# bestepochnumbers[neurons-1] = convergence.best_error_epoch
print("Now saving network")
nn.save_network()
print("Done")

# postprocess.plot_multiple_digits(images_val[ind,:,:], results["prediction"][ind])
# postprocess.plot_images_with_probabilities(images_val[ind,:,:], validationoutput[ind,:])

# print("Total time elapsed: {}".format(totaltime.stop()))

# print(neuronresults)
# np.save("errors.npy", neuronresults)
# np.save("epochs.npy", bestepochtimes)
# np.save("times.npy", bestepochnumbers)
