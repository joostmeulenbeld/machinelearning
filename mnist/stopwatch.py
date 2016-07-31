import time
import datetime

def remaining_time(just_finished, total, time):
    """Calculate the remaining timedelta
    INPUT:
        just_finished: the epoch that was just completed
        total: the total number of epochs that need to be completed
        time: current time of execution
    """
    assert just_finished > 0, "just_finished should equal 1 for the first epoch, not 0"
    epochs_left = total-just_finished
    return time / just_finished * epochs_left

def format_time(t):
    """ Format time t, in seconds, in HH:MM:SS
    """
    return time.strftime("%H:%M:%S", time.gmtime(t))


class Stopwatch(object):
    """ A simple stopwatch; you can start, stop, reset and check the time
    """

    def __init__(self, name="", start=False, t0=0):
        """Initialize the stopwatch
        INPUT:
            name="": the name used for debugging
            start=False: immediately start the stopwatch or not
            t0=0: the start time of the stopwatch
        """
        self.name = name
        self.__tcum = t0
        self.__t = 0
        self.__running = False
        if start:
            self.start()

    def reset(self, start=False):
        t_temp = self.get_time()
        self.__tcum = 0
        self.__t = 0
        self.__running = False
        if start:
            self.start()
        return t_temp

    def start(self):
        assert not self.__running, "Stopwatch {} is already running!".format(self.name)
        self.__running = True
        self.__t = time.time()

    def stop(self):
        assert self.__running, "Stopwatch {} is not running!".format(self.name)
        self.__running = False
        self.__tcum += time.time() - self.__t
        return self.__tcum

    def get_time(self):
        """ Get the elapsed time in seconds (float precision)
        """
        if not self.__running:
            return self.__tcum
        else:
            return self.__tcum + (time.time() - self.__t)

    def get_time_formatted(self, t=None):
        """ Get the elapsed time, formatted by HH:MM:SS
        """
        return format_time(self.get_time())

    def get_remaining_time_formatted(self, just_finished, total):
        return format_time(remaining_time(just_finished, total, self.get_time()))

if __name__ == "__main__":
    sw = Stopwatch(name="joost")
    sw.start()
    time.sleep(2)
    print("Elapsed time (should be plus-minus 2 second): {}".format(sw.get_time()))
    sw.stop()
    time.sleep(1)
    print("Elapsed time (should still be plus-minus 2 second): {}".format(sw.get_time()))
    print("Now formatted: {}".format(sw.get_time_formatted()))

    sw.reset()
    time.sleep(0.1)
    print("This should now be zero: {}".format(sw.get_time()))
