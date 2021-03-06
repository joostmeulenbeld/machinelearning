from time import time
from datetime import timedelta
import itertools
import os
import pickle
os.environ["THEANO_FLAGS"] = "floatX=float32,device=gpu,dnn.enabled=True,optimizer_including=conv_meta,metaopt.verbose=1,optimizer_excluding=more_mem,lib.cnmem=0.75"
                                                                                                                                                                         
import numpy as np

import preprocess
import network
import intraprocess
import postprocess
import stopwatch as sw
from settings import NetworkSettings

# Create a list of 20 integers between 1 and 1000, regularly spaced on log-space
layersizes = list(np.unique(np.floor(10**np.linspace(0.3, 3, 20))).astype(np.int32))

# Create all permutations, up to 3 layers deep
permutations = []
for i in range(4):
    permutations.extend(list(itertools.permutations(layersizes, i)))

neuronresults = [0]*len(permutations)
bestepochnumbers = [0]*len(permutations)
bestepochtimes = [0]*len(permutations)

supertotaltime = sw.Stopwatch()

for (n, layersizes) in enumerate(permutations):
    neuronresultstemp = []
    bestepochtemp = []
    bestepochtimetemp = []
    # Do 5 runs per network to get a good average
    for run in range(5):
        totaltime = sw.Stopwatch(start=True)

        trainsettings = {
            "n_epochs": 1000000,
            "convergence_n_epochs": 300
        }

        networksettings = NetworkSettings(
            networktype="FullyConnected",
            layersizes=layersizes,
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
        convergence = intraprocess.Convergence(trainsettings["convergence_n_epochs"])

        for i in range(1, trainsettings["n_epochs"]+1):

            output, loss = nn.train_fn(images_train, targetoutput)
            validationoutput = nn.eval_fn(images_val)
            results = postprocess.analyze_results(images_val, validationoutput, labels_val)

            convergence.add_point(results["fraction_incorrect"], totaltime.get_time())
            if convergence.converged():
                break

        neuronresultstemp.append(convergence.best_error)
        bestepochtemp.append(convergence.best_error_epoch)
        bestepochtimetemp.append(convergence.time)

        print("\tNeurons: {} run {}, final error: {}. epochs: {}, time elapsed: {}".format(layersizes, run+1, results["fraction_incorrect"], convergence.best_error_epoch, totaltime.stop()))

    neuronresults[n] = np.median(neuronresultstemp)
    bestepochtimes[n] = np.median(bestepochtimetemp)
    bestepochnumbers[n] = np.median(bestepochtemp)

    struct = {
        "permutations": permutations,
        "errors": neuronresults,
        "epochs": bestepochnumbers,
        "times": bestepochtimes
    }
    pickle.dump(struct, open("research.pkl", "wb"))

    print("Estimated remaining time: {}".format(sw.format_time((len(permutations)-n)*int(supertotaltime.reset(start=True)))))
