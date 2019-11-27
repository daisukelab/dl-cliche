from .general import *
import unittest
import random

def recursive_test_array(cls, a, b, msg=None, fn=None):
    """Test corresponding items in two array like items recursively.

    Example:
        Testing list with real numbers which could have very little change caused by arithmetical operation.
        ```python
        def test_arrays_example(self):
            recursive_test_array(self, array1, array2, fn=assertAlmostEqual)
        ```
    """
    if msg is None: msg = ''
    if fn is None:  fn = cls.assertEqual

    if isinstance(a, (list, np.ndarray)):
        for i, (_a, _b) in enumerate(zip(a, b)):
            recursive_test_array(cls, _a, _b, msg=msg+('{},'.format(i)), fn=fn)
    else:
        fn(a, b, msg=msg)


def df_test_exactly_same(title, df1, df2, fillna=0, return_diff=False):
    """Test that two pandas DataFrames are the same, and print result.
    If there's anything different, differences will be shown.

    Arguments:
        title: Title text to print right before result.
        df1: One DataFrame to compare.
        df2: Another DataFrame.
        fillna: Fill N/A with its value beforehand if it is not None.
        return_diff: Returns difference of dfs.
    """
    df1 = df1.fillna(fillna)
    df2 = df2.fillna(fillna)
    diff = []
    try:
        if len(df1) != len(df2):
            raise Exception('DataFrames have different lengths. %d != %d' % (len(df1), len(df2)))
        if len(df1.columns) != len(df2.columns) or not np.all(df1.columns == df2.columns):
            raise Exception('DataFrames have different columns. {} vs. {}'.format(df1.columns, df2.columns))
        result = df1.equals(df2)#np.all(np.all(df1 == df2)) # np.all for rows, and cols -> final answer
        if not result:
            diff = list(np.where(df1 != df2))
            raise Exception('Differences are [rows, columns] = \n{}'.format(diff))
        print(title, 'Passed')
    except Exception as e:
        print(title, 'Failed:', e)
        result = False
    if return_diff:
        return result, diff
    return result


def excel_test_exactly_same(title, excel1, excel2, fillna=0, return_diff=False):
    """Test that two Excel books are the same and print result.
    If there's anything different, differences will be shown.

    Arguments:
        title: Title text to print right before result.
        excel1: One Excel book to compare.
        excel2: Another Excel book.
        fillna: Fill N/A with its value beforehand if it is not None.
        return_diff: Returns difference of dfs.
    """
    dfs1 = df_load_excel_like(excel1, sheetname=None)
    dfs2 = df_load_excel_like(excel2, sheetname=None)
    results = []

    if len(dfs1) != len(dfs2):
        print(f'Failed due to different # of sheets: {excel1}={len(dfs1)}, {excel2}={len(dfs2)}')
        return False

    for k1, k2 in zip(dfs1, dfs2):
        df1 = dfs1[k1]
        df2 = dfs2[k2]
        results.append(test_exactly_same_df(f'{k1} vs {k2} ?', df1, df2, fillna=fillna, return_diff=return_diff))

    single_result = np.all(results)
    print(f'{title} {"Passed" if single_result else "Failed"}')
    return single_result


def create_dummy_file(size, randomize=False, folder=Path('/tmp'), filename=None, suffix=None):
    """Create a dummy file for testing purpose or anything.
    Thanks to https://stackoverflow.com/questions/8816059/create-file-of-particular-size-in-python
    """
    assert size > 1
    folder = Path(folder)
    if not folder.is_dir(): return None

    pathname = folder/(filename or 'foo')
    if suffix is not None: pathname = pathname.with_suffix(suffix)
    stem_len = len(pathname.stem)
    for i in range(100): # retry
        if not pathname.exists(): break
        pathname = pathname.parent/f'{pathname.stem[:stem_len]}{i}{pathname.suffix}'
    if pathname.exists(): return None # cannot determine pathname, all there!

    f = open(pathname, 'wb')
    if randomize is False:
        f.seek(size - 1)
        f.write(b'\0') # Quick solution
    else:
        import string
        contents = bytes(random.choices([ord(s) for s in string.printable], k=size))
        f.write(contents)
    f.close()
    return pathname

