import math
import numpy as np
import pandas as pd


def roundup(x, n=10):
    """Round up x to multiple of n."""
    return int(math.ceil(x / n)) * n


def running_mean(x, N):
    """Calculate running/rolling mean or moving average.
    Thanks to https://stackoverflow.com/a/27681394/6528729
    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def np_describe(arr):
    """Describe numpy array statistics.
    Thanks to https://qiita.com/AnchorBlues/items/051dc69e81705b52adad
    """
    return pd.DataFrame(pd.Series(arr.ravel()).describe()).transpose()


def np_softmax(z):
    """Numpy version softmax.
    Thanjs to https://stackoverflow.com/a/39558290/6528729
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def geometric_mean_preds(list_preds):
    """Calculate geometric mean of prediction results.
    Prediction result is expected to be probability value of ML model's
    softmax outputs in 2d array.

    Arguments:
        list_preds: List of 2d numpy array prediction results.

    Returns:
        Geometric mean of input list of 2d numpy arrays.
    """
    preds = list_preds[0].copy()
    for next_preds in list_preds[1:]:
        preds = np.multiply(preds, next_preds)
    return np.power(preds, 1/len(list_preds))


def arithmetic_mean_preds(list_preds):
    """Calculate arithmetic mean of prediction results.
    Prediction result is expected to be probability value of ML model's
    softmax outputs in 2d array.

    Arguments:
        list_preds: List of 2d numpy array prediction results.

    Returns:
        Arithmetic mean of input list of 2d numpy arrays.
    """
    preds = list_preds[0].copy()
    for next_preds in list_preds[1:]:
        preds = np.add(preds, next_preds)
    return preds / len(list_preds)


class OnlineStats:
    """Calculate mean/variance of a vector online
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
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
