"""
Log test.
"""
import unittest
from dlcliche.utils import *
from dlcliche.math import *
import re

class TesMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_online_stats(self):
        import random
        k = 50
        n = 1000
        a = np.array([[random.normalvariate(_k, 2) for _k in range(k)] for _ in range(n)])
        onstat = OnlineStats(k)

        for _a in a:
            onstat.put(_a)

        def is_in_range(value, amin=-1, amax=1):
            return (amin < value) and (value < amax)

        self.assertTrue(np.all([is_in_range(_1 - _2) for _1, _2 in zip(onstat.mean(), range(k))]))
        self.assertTrue(np.all([is_in_range(v, amin=3, amax=5) for v in onstat.variance()]))
    

if __name__ == '__main__':
    unittest.main()
