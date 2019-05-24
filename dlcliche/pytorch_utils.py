from .utils import *

def pytorch_show_trainable(model):
    """
    Print 'Trainable' or 'Frozen' for all elements in a model.
    Thanks to https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/15
    """
    for name, child in model.named_children():
        for param in child.parameters():
            print('Trainable' if param.requires_grad else 'Frozen', '@', str(child).replace('\n', '')[:80])
        pytorch_show_trainable(child)

