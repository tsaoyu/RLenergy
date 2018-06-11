import numpy as np
from gym.core import Space

class ContinuousSpace(Space):

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        return (self.low < x < self.high)

    def discretize(self, n_discr):
        return np.linspace(self.low, self.high, n_discr).tolist()


