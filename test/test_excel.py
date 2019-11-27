"""
Excel/pandas test.
"""
import unittest
from dlcliche.utils import *
from dlcliche.excel import *
from dlcliche.test import *
from datetime import date

file_folder = Path(__file__).parent

class TestExcel(unittest.TestCase):
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

    def test_df_merge_update(self):
        # Load daily data
        dfs = [df_load_excel_like(file_folder/f'data/{date(2018, 11, d)}.csv').set_index('created')
                                  for d in range(17, 22)]
        # Append to make big one file
        df = df_merge_update(dfs)
        # Resample for every 10 items
        df = df[::10]
        # Load reference
        ref_df = df_load_excel_like(file_folder/'data/ref_merge_resampled.csv').set_index('created')
        # Test
        self.assertTrue(df_test_exactly_same('df_merge_update test', ref_df, df))

if __name__ == '__main__':
    unittest.main()
