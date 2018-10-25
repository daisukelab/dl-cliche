from .general import *
from .ignore_warnings import *

def ensure_folder(folder):
    """Make sure a folder exists."""
    Path(folder).mkdir(exist_ok=True, parent=True)

def write_text_list(textfile, a_list):
    """Write list of str to a file with new lines."""
    with open(textfile, 'w') as f:
        f.write('\n'.join(a_list)+'\n')

from itertools import chain
def flatten_list(lists):
    return list(chain.from_iterable(lists))

def all_elements_are_identical(iterator):
    # https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def get_top_k(top_k, predictions):
    return np.argsort(-predictions, axis=1)[:, :top_k]

def get_top_k_labels(labels, top_k, predictions):
    return np.array(labels)[np.argsort(-predictions, axis=1)[:, :top_k]]
