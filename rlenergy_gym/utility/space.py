import numpy as np
from gym.core import Space

class ContinuousSpace(Space):

    def __init__(self, low, high):
        if low < high:
            self.low = low
            self.high = high
        else:
            raise ValueError('The lower bounds is not smaller than the higher bound.')

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        return (self.low < x < self.high)

    def discretize(self, n_discr):
        return np.linspace(self.low, self.high, n_discr).tolist()


