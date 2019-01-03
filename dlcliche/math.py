import math
import numpy as np


def roundup(x, n=10):
    """Round up x to multiple of n."""
    return int(math.ceil(x / n)) * n


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class OnlineStats:
    """Calculate mean/variance of a vector online
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, length):
        self.K = np.zeros((length))
        self.Ex = np.zeros((length))
        self.Ex2 = np.zeros((length))
        self.n = 0

    def put(self, x):
        if self.n == 0:
            self.K = x
        self.n += 1
        d = x - self.K
        self.Ex += d
        self.Ex2 += d * d

    def undo(self, x):
        self.n -= 1
        d = x - self.K
        self.Ex -= d
        self.Ex2 -= d * d

    def mean(self):
        if self.n == 0:
            return np.zeros_like(self.K)
        return self.K + self.Ex / self.n

    def variance(self):
        if self.n < 2:
            return np.zeros_like(self.K)
        return (self.Ex2 - (self.Ex * self.Ex) / self.n) / (self.n - 1)

    def sigma(self):
        if self.n < 2:
            return np.zeros_like(self.K)
        return np.sqrt(self.variance())

    def count(self):
        return self.n
