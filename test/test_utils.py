import unittest
from dlcliche.utils import *

class TestUtils(unittest.TestCase):
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

    def test_flatten_list(self):
        X = [
            [[[1.0,0.0,0.0]]],
            [[[0.8,0.0,0.1]]],
            [[[0.5,0.5,0.0]]],
            [[[0.4,0.4,0.0]]],
            [[[0.6,0.7,0.0]]],
            [[[0.1,0.5,0.5]]],
        ]
        y1 = [
            [[1.0,0.0,0.0]],
            [[0.8,0.0,0.1]],
            [[0.5,0.5,0.0]],
            [[0.4,0.4,0.0]],
            [[0.6,0.7,0.0]],
            [[0.1,0.5,0.5]],
        ]

        self.assertEqual(y1, flatten_list(X))

if __name__ == '__main__':
    unittest.main()


