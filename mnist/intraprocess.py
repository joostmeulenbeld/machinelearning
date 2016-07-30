import time

import numpy as np
from scipy import stats
import matplotlib
# matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt


class Ploterror(object):
    """ Plot validation and training error. Live plotting is facilitated by imcrementally adding data points
    """

    def __init__(self, trainerrors=[], valerrors=[]):

        assert len(trainerrors)==len(valerrors), "Number of data points should be equal in trainerrors and valerrors"
        plt.ion()
        self.trainerrors = trainerrors
        self.valerrors = valerrors

        self.fig, self.ax = plt.subplots(1, 1)
        # self.ax.hold(True)
        # plt.show()

        self.line_train = self.ax.plot(1+np.arange(len(self.trainerrors)), self.trainerrors, label="Training eror")[0]
        self.line_val = self.ax.plot(1+np.arange(len(self.trainerrors)), self.valerrors, label="Validation error")[0]
        self.draw()

    def add_point(self, trainerror, valerror):
        self.trainerrors.append(trainerror)
        self.valerrors.append(valerror)
        self.draw()

    def add_array(self, trainerrors, valerrors):
        assert len(trainerrors)==len(valerrors), "Number of data points should be equal in trainerrors and valerrors"
        self.trainerrors.extend(trainerrors)
        self.valerrors.extend(valerrors)
        self.draw()

    def draw(self):
        self.line_train.set_data(1+np.arange(len(self.trainerrors)), self.trainerrors)
        self.line_val.set_data(1+np.arange(len(self.valerrors)), self.valerrors)
        # plt.show()
        plt.show()
        plt.pause(0.001)

class Convergence(object):
    """Online analysis of error, calculate if the training is converged
    The most recent n_epochs are checked to see if the lowest error has not been improved
    """

    def __init__(self, n_epochs=50):
        self.errors = np.array([])
        self.n_epochs = n_epochs
        self.best_error = 1
        self.best_error_epoch = 0

    def add_point(self, error):
        """ add a data point
        INPUT:
            error: the new errors
        OUTPUT:
            boolean if training has now converged or not
        """
        self.errors = np.append(self.errors, error)
        if error < self.best_error:
            self.best_error = error
            self.best_error_epoch = self.errors.size
        return self.converged()

    def converged(self):
        """ Return if the current error array is converged or not
        check the last n_epochs if the best error value is within that range
        """
        if self.errors.size < self.n_epochs:
            return False
        return self.errors.size > self.best_error_epoch + self.n_epochs


if __name__ == "__main__":
    pe = Ploterror(trainerrors=[0,2], valerrors=[0,3])
    while True:
        time.sleep(1)
        pe.add_point(np.random.random(), np.random.random())
        print(pe.trainerrors)
