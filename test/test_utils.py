import unittest
from dlcliche.utils import *

class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_flatten_list(self):
        X = [
            [[[1.0,0.0,0.0]]],
            [[[0.8,0.0,0.1]]],
            [[[0.5,0.5,0.0]]],
            [[[0.4,0.4,0.0]]],
            [[[0.6,0.7,0.0]]],
            [[[0.1,0.5,0.5]]],
        ]
        y1 = [
            [[1.0,0.0,0.0]],
            [[0.8,0.0,0.1]],
            [[0.5,0.5,0.0]],
            [[0.4,0.4,0.0]],
            [[0.6,0.7,0.0]],
            [[0.1,0.5,0.5]],
        ]

        self.assertEqual(y1, flatten_list(X))

    def test_pandas_utility(self):
        # df_merge_update
        agg_df = pd.DataFrame({'idx': [1, 2, 3, 4, 5], 'data': [100, 101, 102, 103, 104]})
        agg_df = agg_df.set_index('idx')
        df = pd.DataFrame({'idx': [3, 4, 6, 7], 'data': [110, 111, 112, 113]})
        df = df.set_index('idx')

        agg_df = df_merge_update([agg_df, df])
        reference_df = pd.DataFrame({'data': [100, 101, 110, 111, 104, 112, 113], 'idx': [1, 2, 3, 4, 5, 6, 7]})
        reference_df = reference_df.set_index('idx')
        self.assertTrue(agg_df.equals(reference_df))

        # df_select_by_keyword
        test_df = pd.DataFrame({'one': ['The quick brown fox jumps over the lazy dog', 'Tiny dog'],
                                'two': ['Little bird', 'Sniffing skunk']})
        df = df_select_by_keyword(test_df, 'lazy')
        self.assertTrue(df.equals(df[:1]))
        df = df_select_by_keyword(test_df, 'Little')
        self.assertTrue(df.equals(df[0:]))
        df = df_select_by_keyword(test_df, 'Little', search_columns=['one'])
        self.assertTrue(len(df) == 0)

        # df_str_replace
        df = pd.read_csv('test/data/test_df_op.csv')
        df_str_replace(df, 'ABC', 'HIJ')
        df_str_replace(df, '1', '8')
        df_str_replace(df, '字', '句')
        df_str_replace(df, '.', '-', regex=False)
        df.to_csv('test/data/__test_df_op.csv', index=None)
        pd.read_csv('test/data/ref1_test_df_op.csv').to_csv('test/data/_1_test_df_op.csv', index=None)
        self.assertTrue(df.equals(pd.read_csv('test/data/ref1_test_df_op.csv')))

        df = pd.read_csv('test/data/test_df_op.csv')
        df_str_replace(df, '[Bb]', 'H')
        df_str_replace(df, '4\\.56', '7.77')
        df_str_replace(df, 'def123\\,\\.', 'def123-/')
        df_str_replace(df, '[文字]', '句')
        df.to_csv('test/data/__test_df_op.csv', index=None)
        pd.read_csv('test/data/ref2_test_df_op.csv').to_csv('test/data/_2_test_df_op.csv', index=None)
        self.assertTrue(df.equals(pd.read_csv('test/data/ref2_test_df_op.csv')))

    def test_file_lock(self):
        lock_file_mutex()
        self.assertTrue(is_file_mutex_locked())
        release_file_mutex()
        self.assertFalse(is_file_mutex_locked())

    def test_date(self):
        self.assertEqual(str_to_date('2010/1/23'), datetime.date(2010, 1, 23))
        self.assertEqual(str_to_date('2010/01/23'), datetime.date(2010, 1, 23))
        self.assertEqual(str_to_date('2010-1-2'), datetime.date(2010, 1, 2))
        self.assertEqual(str_to_date('2010-01-02'), datetime.date(2010, 1, 2))

    def test_clf_metrics(self):
        N = 3
        gts = [1, 2, 0, 1, 2]
        results = [np.eye(N=N)[i] for i in [1, 0, 1, 1, 2]]  # One-hot
        f1, recall, precision, acc = calculate_clf_metrics(gts, results)
        self.assertAlmostEqual(f1, 0.5866666666666667)
        self.assertAlmostEqual(recall, 0.6)
        self.assertAlmostEqual(precision, 0.6666666666666666)
        self.assertAlmostEqual(acc, 0.6)

        results = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7]]
        f1, recall, precision, acc = calculate_clf_metrics([1, 0, 1], results)
        self.assertAlmostEqual(f1 + recall + precision + acc, 4.0)

        skewed = skew_bin_clf_preds(results, binary_bias=0.5)
        f1, recall, precision, acc = calculate_clf_metrics([1, 0, 1], skewed)
        self.assertAlmostEqual(f1, 0.8)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(precision, 0.6666666666666666)
        self.assertAlmostEqual(acc, 0.6666666666666666)

        skewed = skew_bin_clf_preds(results, binary_bias=1.5)
        f1, recall, precision, acc = calculate_clf_metrics([1, 0, 1], skewed)
        self.assertAlmostEqual(f1, 0.6666666666666666)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(acc, 0.6666666666666666)

    def test_read_yaml(self):
        data = read_yaml('test/data/yaml_test.yaml')
        sys.stdout.write(str(data))
        self.assertEqual(data['train_data'], '/data/train')
        self.assertEqual(data['work_folder'], 'tmp')
        self.assertEqual(len(data['src_suffixes']), 3)
        self.assertEqual(data['src_suffixes'][0], '.bmp')
        self.assertEqual(data['n_epochs'], 100)
        self.assertEqual(data['valid_pct'], 0.2)
        self.assertEqual(len(data['lr']), 3)
        self.assertEqual(data['lr'][0], 0.003)
        self.assertEqual(data['train_tfm'], None)
        data = read_yaml('test/data/yaml_test.yaml', fix_none=False)
        self.assertEqual(data['train_tfm'], 'None')


if __name__ == '__main__':
    unittest.main()
