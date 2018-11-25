from .general import *
from debtcollector import removals
import unittest

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

def test_exactly_same_df(title, df1, df2, fillna=True, filler=0):
    """Test two pandas DataFrames are the same.
    And shows where differences are.

    Args:
        fillna: Fill N/A beforehand or not. Note that N/A is always False when compared.
        filler: Valid if fillna=True, filling value to feed to pandas fillna().
    """
    df1 = df1.fillna(filler)
    df2 = df2.fillna(filler)
    try:
        if len(df1) != len(df2):
            raise Exception('DataFrames have different lengths. %d != %d' % (len(df1), len(df2)))
        if not np.all(df1.columns == df2.columns):
            raise Exception('DataFrames have different columns. {} vs. {}'.format(df1.columns, df2.columns))
        result = df1.equals(df2)#np.all(np.all(df1 == df2)) # np.all for rows, and cols -> final answer
        if not result:
            raise Exception('Differences are [rows, columns] = \n{}'.format(list(np.where(df1 != df2))))
        print(title, 'Passed')
    except Exception as e:
        print(title, 'Failed:', e)
        result = False
    return result

@removals.remove
def test_exactly_the_same_df(title, df1, df2, fillna=True, filler=0):
    return test_exactly_same_df(title, df1, df2, fillna=fillna, filler=filler)
