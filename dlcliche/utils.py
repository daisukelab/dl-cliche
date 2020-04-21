from .general import *

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from easydict import EasyDict
from tqdm import tqdm_notebook
import shutil
import datetime
import pickle
from collections import Counter
import random
import subprocess
import yaml
import re


## File utilities

def ensure_folder(folder):
    """Make sure a folder exists."""
    Path(folder).mkdir(exist_ok=True, parents=True)


def ensure_delete(folder_or_file):
    anything = Path(folder_or_file)
    if anything.is_dir() and not anything.is_symlink():
        shutil.rmtree(str(folder_or_file))
    elif anything.exists() or anything.is_symlink():
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
    if is_array_like(src):
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


def expand_path(path):
    """Performs `ls` like operation.
    Lists contents in a folder if path is a folder.
    Expands wildcard if path contains wildcard in its name part."""
    path = Path(path)
    d, n = path.parent, path.name
    return list(Path(d).glob(n))


def _copy_with_prefix(file, dest_folder, prefix, symlinks):
    assert file.is_file()
    new_filename = prefix + file.name
    if symlinks:
        symlink_file(file, dest_folder/new_filename)
    else:
        copy_file(file, dest_folder/new_filename)


def copy_with_prefix(files, dest_folder, prefix, symlinks=False):
    """Copy all files to destination folder,
    and new file names will have prefix+original_filename."""
    if not Path(dest_folder).is_dir():
        raise Exception(f'{dest_folder} has to be an existing folder.')
    # ensure files as array-like object
    files = files if is_array_like(files) else [files]
    # expand wild card
    files = [
        f.absolute() for could_be_wild in files for f in expand_path(could_be_wild)
    ]
    # test all files are actually file
    for f in files:
        if not f.is_file(): raise Exception(f'Error: {f} is not a file.')
    # do it
    do_list_item(_copy_with_prefix, files, dest_folder, prefix, symlinks)



def tgz_all(base_dir, files, dest_tgz_path=None, test=True, logger=None):
    """Make .tgz of file/folders relative from base folder.
    This just does:
        cd base_dir && tar czf dest_tgz_path files
        mkdir /tmp/dest_tgz_path.stem && cd /tmp/dest_tgz_path.stem && tar xf dest_tgz_path.absolute()
        cd base_dir && for f in files: diff /tmp/dest_tgz_path.stem/f f
        """
    logger = get_logger() if logger is None else logger
    if len(files) == 0: return None
    if dest_tgz_path is None:
        dest_tgz_path = base_dir/Path(files[0]).with_suffix('.tgz').name

    # zip them
    files = [str(f) for f in files]
    commands = f'cd {base_dir} && tar czf {dest_tgz_path.absolute()} {" ".join(files)}'
    ret, out = exec_commands(commands)
    if ret != 0:
        logger.error(f'Failed with commands: {commands}\n"{out}"')
        return None

    # test zip
    if test:
        test_folder = Path('/tmp')/('dummy_'+dest_tgz_path.stem)
        ensure_delete(test_folder)
        commands = f'mkdir {test_folder} && cd {test_folder} && tar xf {dest_tgz_path.absolute()}'
        ret, out = exec_commands(commands)
        if ret != 0:
            logger.error(f'Test failed with commands: {commands}\n"{out}"\n* {dest_tgz_path} still exists.')
            return None

        failed = False
        for f in files:
            commands = f'cd {base_dir} && diff {test_folder/f} {f}'
            ret, out = exec_commands(commands)
            if ret != 0:
                logger.error(f'Test failed: {commands}\n{out}\n* {dest_tgz_path} still exists.')
                failed = True

        ensure_delete(test_folder)
        if failed:
            return None

    return dest_tgz_path


def chmod_tree_all(tree_root, mode=0o775):
    """Change permission for all the files or directories under the tree_root."""
    for root, dirs, files in os.walk(tree_root):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)


def subsample_files_in_tree(root, filename_pattern, size):
    """
    Sub-sample list of filenames under root folder.
    This ensures to keep sample balance among folders.

    Arguments:
        root: Root folder to search files from.
        filename_pattern: Wildcard pattern like: '*.png'.
        size:
            (0, 1): size to sub-sample; 0.5 for 50%.
            1 or 1.: 100%.
            integer > 1: Number of samples.

    Returns:
        List of sub-sampled files.
        Note that number of files in a folder could be less than size,
        if original number of files is less than size. No oversampling.
    """
    files = []
    folders = [f for f in root.glob('**') if f.is_dir()]
    for folder in folders:
        candidates = [str(f) for f in folder.glob(filename_pattern)]
        n_sample = int(len(candidates) * size) if size < 1. else \
            len(candidates) if int(size) == 1 else min(size, len(candidates))
        if n_sample <= 0: continue
        files.extend(random.sample(candidates, n_sample))
    return files


def copy_subsampled_files(root, dest, wildcard, size, symlinks=False):
    """
    Copy all files that match wildcard under root folder, to the dest folder.
    Note that all files in the sub tree of root folder will be copied.
    Latter found file among the same name files will survive,
    all others will be overwritten.

    Arguments:
        root: Root source folder.
        dest: destination folder.
        wildcard: Wildcard to find files.
        size: Size to subsample, see subsample_files_in_tree() for the detail.
        symlinks: Keeps symbolic links or makes new copy. See shutil.copytree() for the detail.
    """
    files = subsample_files_in_tree(root, wildcard, size=size)
    ensure_folder(dest)
    for f in files:
        copy_any(Path(f).absolute(), dest, symlinks=symlinks)


def save_as_pkl_binary(obj, filename):
    """Save object as pickle binary file.
    Thanks to https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(filename):
    """Load pickle object from file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def read_yaml(file_name, fix_none=True):
    """Read yaml file and set None if str is 'None'."""
    def fix_dict_none(dict_):
        """Fix dict item 'None' as None"""
        for k in dict_:
            if isinstance(dict_[k], dict):
                fix_dict_none(dict_[k])
            elif isinstance(dict_[k], str) and dict_[k] == 'None':
                dict_[k] = None

    with open(file_name) as f:
        yaml_data = yaml.safe_load(f)
    if fix_none:
        fix_dict_none(yaml_data)
    return yaml_data


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


## Multi process utilities

def caller_func_name(level=2):
    """Return caller function name."""
    return sys._getframe(level).f_code.co_name


def _file_mutex_filename(filename):
    return filename or '/tmp/'+Path(caller_func_name(level=3)).stem+'.lock'


def lock_file_mutex(filename=None):
    """Lock file mutex (usually placed under /tmp).
    Note that filename will be created based on caller function name.
    """
    filename = _file_mutex_filename(filename)
    with open(filename, 'w') as f:
        f.write('locked at {}'.format(datetime.datetime.now()))


def release_file_mutex(filename=None):
    """Release file mutex."""
    filename = _file_mutex_filename(filename)
    ensure_delete(filename)


def is_file_mutex_locked(filename=None):
    """Check if file mutex is locked or not."""
    filename = _file_mutex_filename(filename)
    return Path(filename).exists()


def exec_commands(commands):
    """Execute commands with subprocess.Popen.

    Returns:
        Return code, console output as str.
    """
    p = subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    return retval, p.stdout.read().decode() if p.stdout else ''


## Date utilities

def str_to_date(text):
    if '/' in text:
        temp_dt = datetime.datetime.strptime(text, '%Y/%m/%d')
    else:
        temp_dt = datetime.datetime.strptime(text, '%Y-%m-%d')
    return datetime.date(temp_dt.year, temp_dt.month, temp_dt.day)


def get_week_start_end_dates(week_no:int, year=None) -> [datetime.datetime, datetime.datetime]:
    """Get start and end date of an ISO calendar week.
    ISO week starts on Monday, and ends on Sunday.
    
    Arguments:
        week_no: ISO calendar week number
        year: Year to calculate, None will set this year

    Returns:
        [start_date:datetime, end_date:datetime]
    """
    if not year:
        year, this_week, this_day = datetime.datetime.today().isocalendar()
    start_date = datetime.datetime.strptime(f'{year}-W{week_no:02d}-1', "%G-W%V-%u").date()
    end_date = datetime.datetime.strptime(f'{year}-W{week_no:02d}-7', "%G-W%V-%u").date()
    return [start_date, end_date]


def get_this_week_no(date=None):
    """Get ISO calendar week no of given date.
    If date is not given, get for today."""
    if date is None:
        date = datetime.date.today()
    return date.isocalendar()[1]


def get_num_of_weeks(year):
    """Returns number of weeks in a given year.
    Following wikipedia: _'The number of weeks in a given year is equal to the corresponding week number of 28 December.'_
    """
    return get_this_week_no(date=datetime.date(year=year, month=12, day=28))


def daterange(start_date, end_date, inclusive=False):
    """Yield date from start_date until the day before end_date.
    Note that end_date is NOT inclusive.

    Thanks to https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
    """
    days = int((end_date - start_date).days) + (1 if inclusive else 0)
    for n in range(days):
        yield start_date + datetime.timedelta(n)


def add_days_to_date(one_date, days, not_exceed='today'):
    """Add number of days to one_date that doesn't exceed not_exceed.

    Arguments:
        one_date: datetime.datetime date to add days.
        days: Adding number of days, int.
        not_exceed:
            - datetime.datetime date if limiting resulting date,
            - None if nothing to limit,
            - 'today' if limiting to today.

    Returns:
        datetime.datetime date.
    """
    added = one_date + datetime.timedelta(days)
    not_exceed = datetime.datetime.today().date() if not_exceed == 'today' else not_exceed
    if not_exceed is not None and not_exceed < added:
        added = not_exceed
    return added


## List utilities

def write_text_list(textfile, a_list):
    """Write list of str to a file with new lines."""
    with open(textfile, 'w') as f:
        f.write('\n'.join(a_list)+'\n')


def read_text_list(filename) -> list:
    """Read text file splitted as list of texts, stripped."""
    with open(filename) as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]


from itertools import chain
def flatten_list(lists):
    return list(chain.from_iterable(lists))


def is_array_like(item):
    """Check if item is an array-like object."""
    return isinstance(item, (list, set, tuple, np.ndarray))


def is_flat_array(array):
    """Check if array doesn't have array-like object."""
    for item in array:
        if is_array_like(item): return False
    return True


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


import unicodedata
def unicode_visible_width(unistr):
    """Returns the number of printed characters in a Unicode string."""
    return sum([1 if unicodedata.east_asian_width(char) in ['N', 'Na'] else 2 for char in unistr])


def int_from_text(text, default=0):
    """Extract leftmost int number found in given text."""

    g = re.search(r'\d+', str(text))
    return default if g is None else int(g.group(0))


## Pandas utilities

def df_to_csv_excel_friendly(df, filename, **args):
    """df.to_csv() to be excel friendly UTF-8 handling."""
    df.to_csv(filename, encoding='utf_8_sig', **args)


def df_merge_update(df_list):
    """Merge data frames while update duplicated index with followed row.
    
    Usages:
        - df_merge_update([df1, df2, ...]) merges dataframes on the list.
    """
    master = df_list[0]
    for df in df_list[1:]:
        tmp_df = pd.concat([master, df])
        master = tmp_df[~tmp_df.index.duplicated(keep='last')].sort_index()
    return master


from functools import reduce
def df_merge_simply(dataframes):
    """Merge list of data frames into single data frame.
    All data frames are supposed to have the same columns.
    Thanks to https://stackoverflow.com/questions/44327999/python-pandas-merge-multiple-dataframes/44338256
    """
    # Check that all columns are the same
    df0 = dataframes[0]
    for df in dataframes[1:]:
        assert np.all(df0.columns == df.columns)
    # Merge all
    df_merged = reduce(lambda left,right: pd.merge(left, right, how='outer'), dataframes)
    return df_merged


def df_select_by_keyword(source_df, keyword, search_columns=None, as_mask=False):
    """Select data frame rows by a search keyword.
    Any row will be selected if any of its search columns contain the keyword.
    
    Returns:
        New data frame where rows have the keyword,
        or mask if as_mask is True.
    """
    search_columns = search_columns or source_df.columns
    masks = np.column_stack([source_df[col].str.contains(keyword, na=False) for col in search_columns])
    mask = masks.any(axis=1)
    if as_mask:
        return mask
    return source_df.loc[mask]


def df_select_by_keywords(source_df, keys_cols, and_or='or', as_mask=False):
    """Multi keyword version of df_select_by_keyword.

    Arguments:
        key_cols: dict defined as `{'keyword1': [search columns] or None, ...}`
    """
    masks = []
    for keyword in keys_cols:
        columns = keys_cols[keyword]
        mask = df_select_by_keyword(source_df, keyword, search_columns=columns, as_mask=True)
        masks.append(mask)
    mask = np.column_stack(masks).any(axis=1) if and_or == 'or' else \
           np.column_stack(masks).all(axis=1)
    if as_mask:
        return mask
    return source_df.loc[mask]


def df_mask_by_str_or_list(df, column, keys):
    """Find str match and make mask of dataframe.
    If multiple keys are fed, mask will be AND-calculated among keys.
    """
    mask = None
    if type(keys) == str: keys = [keys]
    for key in keys:
        this_mask = df[column].str.find(key) >= 0
        mask = this_mask if mask is None else (mask & this_mask)
    return mask


def df_mask_by_str_conditions(df, conditions):
    """Find dataframe rows that matches condition of str search.
    Returns:
        Aggregated mask from masks calculated from sub conditions recursively.
    """
    col_or_op, key_or_conds = conditions
    if is_array_like(key_or_conds):
        if col_or_op not in ['and', 'or']:
            raise Exception(f'unknown condition: {col_or_op}')
        masks = [df_mask_by_str_conditions(df, sub_conds) for sub_conds in key_or_conds]
        mask = np.column_stack(masks).any(axis=1) if col_or_op == 'or' else \
               np.column_stack(masks).all(axis=1)
        return mask
    else:
        return df_mask_by_str_or_list(df, col_or_op, key_or_conds)


def df_str_replace(df, from_strs, to_str, regex=True):
    """Apply str.replace to entire DataFrame inplace.

    - All string columns will be applied. (dtype == 'objet')
    - All other dtype columns will not be applied.
    """
    for c in df.columns:
        if df[c].dtype != 'object': continue
        df[c] = df[c].str.replace(from_strs, to_str, regex=regex)


def df_cell_str_replace(df, from_str, to_str):
    """Replace cell string with new string if entire string matches."""
    df_str_replace(df, from_strs, to_str, regex=False)


def df_print_differences(df1, df2):
    """Print all difference between two dataframes."""
    if df1.shape != df2.shape:
        print(f'Error: df1.shape={df1.shape} != df2.shape{df2.shape}')
        return
    rows, cols = np.where(df1 != df2)
    for r, c in zip(rows, cols):
        print(f'at[{r},{c}] "{df1.iat[r, c]}" != "{df2.iat[r, c]}"')


_EXCEL_LIKE = ['.csv', '.xls', '.xlsx', '.xlsm']
def is_excel_file(filename):
    # not accepted if suffix == '.csv': return True
    return Path(filename).suffix.lower() in _EXCEL_LIKE


def is_csv_file(filename):
    return Path(filename).suffix.lower() == '.csv'


def pd_read_excel_keep_dtype(io, **args):
    """pd.read_excel() wrapper to do as described in pandas document:
    '... preserve data as stored in Excel and not interpret dtype'

    Details:
        - String '1' might be loaded as int 1 by pd.read_excel(file).
        - By setting `dtype=object` it will preserve it as string '1'.
    """
    return pd.read_excel(io, dtype=object, **args)


def pd_read_csv_as_str(filename, **args):
    """pd.read_csv() wrapper to preserve data type = str"""
    return pd.read_csv(filename, dtype=object, **args)


def df_load_excel_like(filename, preserve_dtype=True, **args):
    """Load Excel like files. (csv, xlsx, ...)"""
    if is_csv_file(filename):
        if preserve_dtype:
            return pd_read_csv_as_str(filename, **args)
        return pd.read_csv(filename, **args)
    if preserve_dtype:
        return pd_read_excel_keep_dtype(filename, **args)
    return pd.read_excel(filename, **args)


import codecs
def df_read_sjis_csv(filename, **args):
    """Read shift jis Japanese csv file.
    Thanks to https://qiita.com/niwaringo/items/d2a30e04e08da8eaa643
    """
    with codecs.open(filename, 'r', 'Shift-JIS', 'ignore') as file:
        return pd.read_table(file, delimiter=',', **args)


def df_highlight_max(df, color='yellow', axis=1):
    """Highlight max valued cell with color.
    Thanks to https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    """
    def highlight_max(s):
        '''Highlight the maximum in a Series yellow or any color.'''
        is_max = s == s.max()
        return [f'background-color: {color}' if v else '' for v in is_max]
    df = df.copy()
    return df.style.apply(highlight_max, axis=axis)


def df_apply_sns_color_map(df, color='red', **kwargs):
    """Set color map to a dataframe.
    Thanks to https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    """
    import seaborn as sns
    cm = sns.light_palette(color, as_cmap=True, **kwargs)
    df = df.copy()
    return df.style.background_gradient(cmap=cm)


## Dataset utilities

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
    from imblearn.over_sampling import RandomOverSampler
    return  _balance_class(X, y, 'max', RandomOverSampler, random_state)


def balance_class_by_under_sampling(X, y, random_state=42):
    """Balance class distribution with imbalanced-learn RandomUnderSampler."""
    from imblearn.under_sampling import RandomUnderSampler
    return  _balance_class(X, y, 'min', RandomUnderSampler, random_state)


def df_balance_class_by_over_sampling(df, label_column, random_state=42):
    """Balance class distribution in DataFrame with imbalanced-learn RandomOverSampler."""
    X, y = list(range(len(df))), list(df[label_column])
    X, _ = balance_class_by_over_sampling(X, y, random_state=random_state)
    return df.iloc[X].sort_index()


def df_balance_class_by_under_sampling(df, label_column, random_state=42):
    """Balance class distribution in DataFrame with imbalanced-learn RandomUnderSampler."""
    X, y = list(range(len(df))), list(df[label_column])
    X, _ = balance_class_by_under_sampling(X, y, random_state=random_state)
    return df.iloc[X].sort_index()


def balance_class_by_limited_over_sampling(X, y, max_sample_per_class=None, multiply_limit=2., random_state=42):
    """Balance class distribution basically by oversampling but limited duplication.

    # Arguments
        X: Data samples, only size of samples is used here.
        y: Class labels to be balanced.
        max_sample_per_class: Number of maximum samples per class, large class will be limitd to this number.
        multiply_limit: Small size class samples will be duplicated, but limited to multiple of this number.
    """
    assert len(X) == len(y), f'Length of X({len(X)}) and y({len(y)}) is different, supposed to be the same.'
    y_count = Counter(y)
    max_sample_per_class = max_sample_per_class or np.max(list(y_count.values()))
    resampled_idxes = []
    random.seed(random_state)
    for cur_y, count in y_count.items():
        this_samples = np.min([multiply_limit * count, max_sample_per_class]).astype(int)
        idxes = np.where(y == cur_y)[0]
        # Add all class samples first
        resampled_idxes += list(idxes)
        # Add oversampled
        idxes = random.choices(idxes, k=this_samples-len(idxes))
        resampled_idxes += list(idxes)
    return X[resampled_idxes], y[resampled_idxes]


def df_balance_class_by_limited_over_sampling(df, label_column,
                                              max_sample_per_class=None, multiply_limit=2.,
                                              random_state=42):
    """Balance class distribution in DataFrame with balance_class_by_limited_over_sampling."""
    X, y = np.array(range(len(df))), df[label_column].values
    X, _ = balance_class_by_limited_over_sampling(X, y, max_sample_per_class=max_sample_per_class,
                                                  multiply_limit=multiply_limit, random_state=random_state)
    return df.iloc[X].sort_index()


from sklearn.model_selection import train_test_split

def subsample_stratified(X, y, size=0.1):
    """
    Stratified subsampling.
    """
    _, X_test, _, y_test = train_test_split(X, y, test_size=size, stratify=y)
    return X_test, y_test


## Visualization utilities

def _expand_labels_from_y(y, labels):
    """Make sure y is index of label set."""
    if labels is None:
        labels = sorted(list(set(y)))
        y = [labels.index(_y) for _y in y]
    return y, labels


def visualize_class_balance(title, y, labels=None, sorted=False):
    y, labels = _expand_labels_from_y(y, labels)
    sample_dist_list = get_class_distribution_list(y, len(labels))
    if sorted:
        items = list(zip(labels, sample_dist_list))
        items.sort(key=lambda x:x[1], reverse=True)
        sample_dist_list = [x[1] for x in items]
        labels = [x[0] for x in items]
    index = range(len(labels))
    fig, ax = plt.subplots(1, 1, figsize = (16, 5))
    ax.bar(index, sample_dist_list)
    ax.set_xlabel('Label')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    fig.show()


from collections import OrderedDict
def print_class_balance(title, y, labels=None, sorted=False):
    y, labels = _expand_labels_from_y(y, labels)
    distributions = get_class_distribution(y)
    dist_dic = {labels[cls]:distributions[cls] for cls in distributions}
    if sorted:
        items = list(dist_dic.items())
        items.sort(key=lambda x:x[1], reverse=True)
        dist_dic = OrderedDict(items) # sorted(dist_dic.items(), key=...) didn't work for some reason...
    print(title, '=', dist_dic)
    zeroclasses = [label for i, label in enumerate(labels) if i not in distributions.keys()]
    if 0 < len(zeroclasses):
        print(' 0 sample classes:', zeroclasses)


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def calculate_clf_metrics(y_true, y_pred, average='weighted'):
    """Calculate metrics: f1/recall/precision/accuracy.

    # Arguments
        y_true: GT, an index of label or one-hot encoding format.
        y_pred: Prediction output, index or one-hot.
        average: `average` parameter passed to sklearn.metrics functions.

    # Returns
        Four metrics: f1, recall, precision, accuracy.
    """
    y_true = flatten_y_if_onehot(y_true)
    y_pred = flatten_y_if_onehot(y_pred)
    if np.max(y_true) < 2 and np.max(y_pred) < 2:
        average = 'binary'

    f1 = f1_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, recall, precision, accuracy


def skew_bin_clf_preds(y_pred, binary_bias=None, logger=None):
    """Apply bias to prediction results for binary classification.
    Calculated as follows.
        p(y=1) := p(y=1) ^ binary_bias
        p(y=0) := 1 - p(y=0)
    0 < binary_bias < 1 will be optimistic with result=1.
    Inversely, 1 < binary_bias will make results pesimistic.
    """
    _preds = np.array(y_pred.copy())
    if binary_bias is not None:
        ps = np.power(_preds[:, 1], binary_bias)
        _preds[:, 1] = ps
        _preds[:, 0] = 1 - ps
        logger = get_logger() if logger is None else logger
        logger.info(f' @skew{"+" if binary_bias >= 0 else ""}{binary_bias}')
    return _preds


def print_clf_metrics(y_true, y_pred, average='weighted', binary_bias=None, title_prefix='', logger=None):
    """Calculate and print metrics: f1/recall/precision/accuracy.
    See calculate_clf_metrics() and skew_bin_clf_preds() for more detail.
    """
    # Add bias if binary_bias is set
    _preds = skew_bin_clf_preds(y_pred, binary_bias, logger=logger)
    # Calculate metrics
    f1, recall, precision, acc = calculate_clf_metrics(y_true, _preds, average=average)
    logger = get_logger() if logger is None else logger
    logger.info('{0:s}F1/Recall/Precision/Accuracy = {1:.4f}/{2:.4f}/{3:.4f}/{4:.4f}' \
          .format(title_prefix, f1, recall, precision, acc))


# Thanks to https://qiita.com/knknkn1162/items/be87cba14e38e2c0f656
def plt_japanese_font_ready():
    """Set font family with Japanese fonts.
    
    # How to install fonts:
        wget https://ipafont.ipa.go.jp/old/ipafont/IPAfont00303.php
        mv IPAfont00303.php IPAfont00303.zip
        unzip -q IPAfont00303.zip
        sudo cp IPAfont00303/*.ttf /usr/share/fonts/truetype/
    """
    plt.rcParams['font.family'] = 'IPAPGothic'


def plt_looks_good():
    """Plots will be looks good (at least to me)."""
    plt.rcParams["figure.figsize"] = [16, 10]
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def pd_display_more(max_cols=100, max_rows=500):
    """Set max cols/rows of pandas display."""
    pd.options.display.max_columns = max_cols
    pd.options.display.max_rows = max_rows


# Thanks to http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """Plot confusion matrix."""
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


def deterministic_everything(seed=42, pytorch=True, tf=False):
    """Set pseudo random everything deterministic. a.k.a. `seed_everything`
    Universal to major frameworks.

    Thanks to https://docs.fast.ai/dev/test.html#getting-reproducible-results
    Thanks to https://pytorch.org/docs/stable/notes/randomness.html
    """

    # Python RNG
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Numpy RNG
    import numpy as np
    np.random.seed(seed)

    # Pytorch RNGs
    if pytorch:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # TensorFlow RNG
    if tf:
        import tensorflow as tf
        tf.set_random_seed(seed)


def simply_ignore(warnings_):
    """Set warnings to ignore all.

    Usage:
        import warnings; simply_ignore(warnings)
    """
    warnings_.simplefilter('ignore')

