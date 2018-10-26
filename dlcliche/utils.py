from .general import *
from .ignore_warnings import *

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from easydict import EasyDict
from tqdm import tqdm_notebook

def ensure_folder(folder):
    """Make sure a folder exists."""
    Path(folder).mkdir(exist_ok=True, parent=True)

import shutil
def copy_file(src, target):
    """Copy source file to target file."""
    assert Path(src).is_file()
    shutil.copy(str(src), str(target))

def write_text_list(textfile, a_list):
    """Write list of str to a file with new lines."""
    with open(textfile, 'w') as f:
        f.write('\n'.join(a_list)+'\n')

from itertools import chain
def flatten_list(lists):
    return list(chain.from_iterable(lists))

# Thanks to https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_elements_are_identical(iterator):
    """Check all elements in iterable like list are identical."""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

# Thanks to https://github.com/dsindex/blog/wiki/%5Bpython%5D-difflib,-show-differences-between-two-strings
import difflib
def show_text_diff(text, n_text):
    """
    http://stackoverflow.com/a/788780
    Unify operations between two compared strings seqm is a difflib.
    SequenceMatcher instance whose a & b are strings
    """
    seqm = difflib.SequenceMatcher(None, text, n_text)
    output= []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'equal':
            pass # output.append(seqm.a[a0:a1])
        elif opcode == 'insert':
            output.append("<INS>" + seqm.b[b0:b1] + "</INS>")
        elif opcode == 'delete':
            output.append("<DEL>" + seqm.a[a0:a1] + "</DEL>")
        elif opcode == 'replace':
            # seqm.a[a0:a1] -> seqm.b[b0:b1]
            output.append("<REPL>" + seqm.b[b0:b1] + "</REPL>")
        else:
            raise RuntimeError
    return ''.join(output)

## Pandas utilities

def df_to_csv_excel_friendly(df, filename):
    """df.to_csv() to be excel friendly UTF-8 handling."""
    df.to_csv(filename, encoding='utf_8_sig')

