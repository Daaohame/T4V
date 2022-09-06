import time

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_normal_distribution_and_squared_loss`"""
        self.times = []
        self.start()
    def start(self):
        """Start the timer."""
        self.tik = time.time()
    def record(self):
        """Pause the timer and record the elapsed time in a list."""
        self.times.append(time.time() - self.tik)
        # self.tik = time.time()
        return self.times[-1]
    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)
    def sum(self):
        """Return the sum of time."""
        return sum(self.times)
    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()