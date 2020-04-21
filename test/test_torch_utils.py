"""
Torch utils tests.
"""

import unittest
from dlcliche.utils import *
from dlcliche.torch_utils import *
import torch


class TestTorchUtils(unittest.TestCase):
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

    def test_flooding_wrapper(self):
        preds = torch.Tensor([[0.2, 0.5], [0.2, 0.8]])
        y_true = torch.Tensor([1, 0]).to(dtype=torch.long)

        normal_loss = nn.CrossEntropyLoss()
        normal = normal_loss(preds, y_true)

        b = 0.8
        crit = LossFlooding(nn.CrossEntropyLoss(), b)
        flooded = crit(preds, y_true)
        self.assertTrue((flooded - b)  == (b - normal))

        b = 0.7
        crit = LossFlooding(nn.CrossEntropyLoss(), b)
        flooded = crit(preds, y_true)
        self.assertTrue(flooded == normal)
        self.assertTrue(b < normal)


if __name__ == '__main__':
    unittest.main()
