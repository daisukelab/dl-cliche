from .general import *
from .ignore_warnings import *

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from easydict import EasyDict
from tqdm import tqdm_notebook
import shutil

def ensure_folder(folder):
    """Make sure a folder exists."""
    Path(folder).mkdir(exist_ok=True, parents=True)

def ensure_delete(folder_or_file):
    anything = Path(folder_or_file)
    if anything.is_dir():
        shutil.rmtree(str(folder_or_file))
    else:
        anything.unlink()

def copy_file(src, dst):
    """Copy source file to destination file."""
    assert Path(src).is_file()
    shutil.copy(str(src), str(dst))

def copy_any(src, dst, symlinks=True):
    """Copy any file or folder recursively."""
    if Path(src).is_dir():
        shutil.copytree(src, dst, symlinks=symlinks)
    else:
        copy_file(src, dst)

def symlink_file(fromfile, tofile):
    """Make fromfile's symlink as tofile."""
    Path(tofile).symlink_to(fromfile)

def convert_mono_to_jpg(fromfile, tofile):
    """Convert monochrome image to color jpeg format.
    Linear copy to RGB channels. 
    https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html

    Args:
        fromfile: float png mono image
        tofile: RGB color jpeg image.
    """
    img = np.array(Image.open(fromfile))
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-4)
    img = (img * 255).astype(np.uint8) # [0,1) float to [0,255] uint8
    img = np.repeat(img[..., np.newaxis], 3, axis=-1) # mono to RGB color
    img = Image.fromarray(img)
    tofile = Path(tofile)
    img.save(tofile.with_suffix('.jpg'), 'JPEG', quality=100)

def make_copy_to(dest_folder, files, n_sample=None, operation=copy_file):
    """Do file copy like operation from files to dest_folder.
    
    If n_sample is set, it creates symlinks up to number of n_sample files.
    If n_sample is greater than len(files), symlinks are repeated twice or more until it reaches to n_sample.
    If n_sample is less than len(files), n_sample symlinks are created for the top n_sample samples in files."""
    dest_folder.mkdir(exist_ok=True, parents=True)
    if n_sample is None:
        n_sample = len(files)

    _done = False
    _dup = 0
    _count = 0
    while not _done: # yet
        for f in files:
            f = Path(f)
            name = f.stem+('_%d'%_dup)+f.suffix if 0 < _dup else f.name
            to_file = dest_folder / name
            operation(f, to_file)
            _count += 1
            _done = n_sample <= _count
            if _done: break
        _dup += 1
    print('Now', dest_folder, 'has', len(list(dest_folder.glob('*'))), 'files.')

## List utilities

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

## Text utilities

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

