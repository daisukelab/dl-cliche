from .general import *
from .ignore_warnings import *

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from easydict import EasyDict
from tqdm import tqdm_notebook
import shutil

## File utilities

def ensure_folder(folder):
    """Make sure a folder exists."""
    Path(folder).mkdir(exist_ok=True, parents=True)

def ensure_delete(folder_or_file):
    anything = Path(folder_or_file)
    if anything.is_dir():
        shutil.rmtree(str(folder_or_file))
    elif anything.exists():
        anything.unlink()

def copy_file(src, dst):
    """Copy source file to destination file."""
    assert Path(src).is_file()
    shutil.copy(str(src), str(dst))

def _copy_any(src, dst, symlinks):
    if Path(src).is_dir():
        if Path(dst).is_dir():
            dst = Path(dst)/Path(src).name
        assert not Path(dst).exists()
        shutil.copytree(src, dst, symlinks=symlinks)
    else:
        copy_file(src, dst)

def copy_any(src, dst, symlinks=True):
    """Copy any file or folder recursively.
    Source file can be list/array of files.
    """
    do_list_item(_copy_any, src, dst, symlinks)

def do_list_item(func, src, *prms):
    if isinstance(src, (list, tuple, np.ndarray)):
        result = True
        for element in src:
            result = do_list_item(func, element, *prms) and result
        return result
    else:
        return func(src, *prms)

def _move_file(src, dst):
    shutil.move(str(src), str(dst))

def move_file(src, dst):
    """Move source file to destination file/folder.
    Source file can be list/array of files.
    """
    do_list_item(_move_file, src, dst)

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

## Log utilities

import logging
_loggers = {}
def get_logger(name=None, level=logging.DEBUG, format=None, print=True, output_file=None):
    """One liner to get logger.
    See test_log.py for example.
    """
    name = name or __name__
    if _loggers.get(name):
        return _loggers.get(name)
    else:
        log = logging.getLogger(name)
    formatter = logging.Formatter(format or '%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s')
    def add_handler(handler):
        handler.setFormatter(formatter)
        handler.setLevel(level)
        log.addHandler(handler)
    if print:
        add_handler(logging.StreamHandler())
    if output_file:
        ensure_folder(Path(output_file).parent)
        add_handler(logging.FileHandler(output_file))
    log.setLevel(level)
    log.propagate = False
    _loggers[name] = log
    return log

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

def df_merge_update(to_df, joining_from_df):
    """Merge data frames while update duplicated index with joining row."""
    tmp_df = pd.concat([to_df, joining_from_df])
    to_df = tmp_df[~tmp_df.index.duplicated(keep='last')].sort_index()
    return to_df

def df_select_by_keyword(source_df, keyword, search_columns=None):
    """Select data frame rows by a search keyword.
    
    Returns:
        New data frame where rows have the keyword.
    """
    search_columns = search_columns or source_df.columns
    mask = np.column_stack([source_df[col].str.contains(keyword, na=False) for col in search_columns])
    return source_df.loc[mask.any(axis=1)]

def pd_read_excel_keep_dtype(io, **args):
    """pd.read_excel() wrapper to do as described in pandas document:
    '... preserve data as stored in Excel and not interpret dtype'

    Details:
        - String '1' might be loaded as int 1 by pd.read_excel(file).
        - By setting `dtype=object` it will preserve it as string '1'.
    """
    return pd.read_excel(io, dtype=object, **args)

## Dataset utilities

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def flatten_y_if_onehot(y):
    """De-one-hot y, i.e. [0,1,0,0,...] to 1 for all y."""
    return y if len(np.array(y).shape) == 1 else np.argmax(y, axis = -1)

def get_class_distribution(y):
    """Calculate number of samples per class."""
    # y_cls can be one of [OH label, index of class, class label name string]
    # convert OH to index of class
    y_cls = flatten_y_if_onehot(y)
    # y_cls can be one of [index of class, class label name]
    classset = sorted(list(set(y_cls)))
    sample_distribution = {cur_cls:len([one for one in y_cls if one == cur_cls]) for cur_cls in classset}
    return sample_distribution

def get_class_distribution_list(y, num_classes):
    """Calculate number of samples per class as list"""
    dist = get_class_distribution(y)
    assert(y[0].__class__ != str) # class index or class OH label only
    list_dist = np.zeros((num_classes))
    for i in range(num_classes):
        if i in dist:
            list_dist[i] = dist[i]
    return list_dist

def _balance_class(X, y, min_or_max, sampler_class, random_state):
    """Balance class distribution with sampler_class."""
    y_cls = flatten_y_if_onehot(y)
    distribution = get_class_distribution(y_cls)
    classes = list(distribution.keys())
    counts  = list(distribution.values())
    nsamples = np.max(counts) if min_or_max == 'max' \
          else np.min(counts)
    flat_ratio = {cls:nsamples for cls in classes}
    Xidx = [[xidx] for xidx in range(len(X))]
    sampler_instance = sampler_class(ratio=flat_ratio, random_state=random_state)
    Xidx_resampled, y_cls_resampled = sampler_instance.fit_sample(Xidx, y_cls)
    sampled_index = [idx[0] for idx in Xidx_resampled]
    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])

def balance_class_by_over_sampling(X, y, random_state=42):
    """Balance class distribution with imbalanced-learn RandomOverSampler."""
    return  _balance_class(X, y, 'max', RandomOverSampler, random_state)

def balance_class_by_under_sampling(X, y, random_state=42):
    """Balance class distribution with imbalanced-learn RandomUnderSampler."""
    return  _balance_class(X, y, 'min', RandomUnderSampler, random_state)

def visualize_class_balance(title, y, labels):
    sample_dist_list = get_class_distribution_list(y, len(labels))
    index = range(len(labels))
    fig, ax = plt.subplots(1, 1, figsize = (16, 5))
    ax.bar(index, sample_dist_list)
    ax.set_xlabel('Label')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    fig.show()

def print_class_balance(title, y, labels):
    distributions = get_class_distribution(y)
    dist_dic = {labels[cls]:distributions[cls] for cls in distributions}
    print(title, '=', dist_dic)
    zeroclasses = [label for i, label in enumerate(labels) if i not in distributions.keys()]
    if 0 < len(zeroclasses):
        print(' 0 sample classes:', zeroclasses)

## Visualization utilities

# Thanks to http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """Plot confusion matrix.
    """
    po = np.get_printoptions()
    np.set_printoptions(precision=2)

    y_test = flatten_y_if_onehot(y_test)
    y_pred = flatten_y_if_onehot(y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if title is None: title = 'Normalized confusion matrix'
    else:
        if title is None: title = 'Confusion matrix (not normalized)'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    np.set_printoptions(**po)
