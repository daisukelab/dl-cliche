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
        flooded = crit(preds, y_true, train=True)
        self.assertTrue((flooded - b)  == (b - normal))

        b = 0.7
        crit = LossFlooding(nn.CrossEntropyLoss(), b)
        flooded = crit(preds, y_true, train=True)
        self.assertTrue(flooded == normal)
        self.assertTrue(b < normal)

        b = 0.8
        crit = LossFlooding(nn.CrossEntropyLoss(), b)
        flooded = crit(preds, y_true, train=False)
        self.assertTrue(flooded == normal)
        self.assertTrue(b > normal)

    def test_mixup(self):
        N, C, H, W = 32, 3, 5, 10
        # test data and fake preds
        inputs = np.arange(N * C * H * W).reshape((N, C, H, W))
        inputs = torch.tensor(inputs)
        targets = np.arange(N)
        targets = torch.tensor(targets)
        preds = np.linspace(1.0, 0.0, num=N)
        preds = [np.roll(preds, i) for i in range(N)]
        preds = torch.tensor(preds)
        # reference loss function
        ref_crit = nn.CrossEntropyLoss(reduction='none')
        # test both train/valid
        for train, alpha in zip([True, True, True, False], [1.0, 0.5, 0.0, 0.0]):
            for _ in range(1000):
                batch_mixer = IntraBatchMixup(nn.CrossEntropyLoss(), alpha=alpha)
                tfmed_inputs, stacked_targets = batch_mixer.transform(inputs, targets, train=train)
                loss = batch_mixer.criterion(preds, stacked_targets)

                org_targets, counterpart_targets, lambd = stacked_targets
                shape = [lambd.size(0)] + [1 for _ in range(len(inputs.shape) - 1)]
                in_lambd = lambd.view(shape)

                calculated_inputs = inputs * in_lambd + inputs[counterpart_targets] * (1-in_lambd)
                in_shape = [lambd.size(0)] + [1 for _ in range(len(inputs.shape) - 1)]
                lambd = lambd.view(in_shape)
                calculated_loss = ref_crit(preds, org_targets) * lambd + ref_crit(preds, counterpart_targets) * (1 - lambd)
                calculated_loss = calculated_loss.mean()

                self.assertTrue(np.all(0.0 <= lambd.numpy()) and np.all(lambd.numpy() <= 1.0))
                self.assertTrue(np.all(calculated_inputs.numpy() == tfmed_inputs.numpy()))
                self.assertTrue(np.all(calculated_loss.numpy() == loss.numpy())), f'{calculated_loss} != {loss}'

    def test_label_smoothing(self):
        logits = torch.Tensor([[0.3529, 0.8618, 0.8859, 0.9957, 0.5551, 0.8189],
                               [0.0897, 0.1646, 0.7691, 0.6098, 0.6384, 0.7858],
                               [0.8170, 0.7350, 0.3148, 0.3416, 0.6053, 0.9115],
                               [0.4621, 0.3512, 0.7762, 0.8315, 0.4907, 0.3521]])
        targets = torch.Tensor([1, 0, 5, 5]).to(torch.long)
        refs = [1.8686657, 1.8686657, 1.8635569, 7.4337926]
        answers = [crit(logits, targets).numpy() for crit in [nn.CrossEntropyLoss(),
                                                              LabelSmoothing(smoothing=0.0),
                                                              LabelSmoothing(smoothing=0.1),
                                                              LabelSmoothing(smoothing=0.2, reduction='sum')]]
        self.assertTrue(np.allclose(refs, answers))

        refs = [1.9255728, 1.9255728, 1.9578586, [1.7212944, 1.5842859, 1.9641012]]
        answers = [crit(logits, targets).numpy() for crit in [nn.CrossEntropyLoss(ignore_index=1),
                                                              LabelSmoothing(smoothing=0.0, ignore_index=1),
                                                              LabelSmoothing(smoothing=0.1, ignore_index=5),
                                                              LabelSmoothing(smoothing=0.2, ignore_index=0, reduction='none')]]
        self.assertTrue(np.allclose(refs[:3], answers[:3]))
        self.assertTrue(np.allclose(refs[3], answers[3]))


if __name__ == '__main__':
    unittest.main()
