from .general import *
import unittest

def recursive_test_array(cls, a, b, msg=None, fn=None):
    """Test corresponding items in two array like items recursively.

    Example:
        Testing list with real numbers which could have very little change caused by arithmetical operation.
        ```python
        def test_arrays_example(self):
            recursive_test_array(self, array1, array2, assertAlmostEqual)
        ```
    """
    if msg is None: msg = ''
    if fn is None:  fn = cls.assertEqual

    if isinstance(a, (list, np.ndarray)):
        for i, (_a, _b) in enumerate(zip(a, b)):
            recursive_test_array(cls, _a, _b, msg=msg+('{},'.format(i)), fn=fn)
    else:
        fn(a, b, msg=msg)

