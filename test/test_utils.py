import unittest
from dlcliche.utils import *

class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_folder = Path('_tmp')
        ensure_folder(cls.tmp_folder)

    @classmethod
    def tearDownClass(cls):
        ensure_delete(cls.tmp_folder)

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

    def make_folder_for_sure(self, folder):
        ensure_delete(folder)
        ensure_folder(folder)
        self.assertTrue(folder.is_dir())

    def make_src_files_for_copy_tests(self, src_folder):
        self.make_folder_for_sure(src_folder)
        for i in range(5):
            sub_folder = src_folder/('sub%d' % i)
            self.make_folder_for_sure(sub_folder)
        for i in range(5):
            sub_file = src_folder/('file%d.txt' % i)
            f = sub_file.open('w')
            f.write('abcde%d' % i)
            f.close()
        return sorted(list(src_folder.iterdir()))

    def test_1_make_copy_to(self):
        self.assertTrue(self.tmp_folder.is_dir())
        copy_src = self.tmp_folder/'src'
        dst = self.tmp_folder/'dst'
        src_all = self.make_src_files_for_copy_tests(copy_src)

        self.make_folder_for_sure(dst)
        make_copy_to(dst, src_all, operation=symlink_file)
        for i in range(5):
            sub_folder = dst/('sub%d' % i)
            self.assertTrue(sub_folder.is_symlink())
        for i in range(5):
            sub_file = dst/('file%d.txt' % i)
            self.assertTrue(sub_file.is_symlink())

        self.make_folder_for_sure(dst)
        make_copy_to(dst, src_all, operation=copy_any)
        for i in range(5):
            sub_folder = dst/('sub%d' % i)
            self.assertTrue(sub_folder.is_dir())
        for i in range(5):
            f = (dst/('file%d.txt' % i)).open()
            text = f.read()
            f.close()
            self.assertEqual('abcde%d' % i, text)

    def test_2_copy_move_single(self):
        self.assertTrue(self.tmp_folder.is_dir())
        copy_src = self.tmp_folder/'src'
        dst = self.tmp_folder/'dst'
        src_all = self.make_src_files_for_copy_tests(copy_src)

        for shift, fn in zip([0, 1], [copy_any, move_file]):
            # clean up
            self.make_folder_for_sure(dst)
            # test single source file
            fn(src_all[5+shift], str(dst))
            self.assertTrue((dst/src_all[5+shift].name).is_dir())
            fn(src_all[shift], str(dst))
            f = (dst/('file%d.txt'%shift)).open()
            text = f.read()
            f.close()
            self.assertEqual('abcde%d'%shift, text)

    def test_3_copy_move_multi(self):
        self.assertTrue(self.tmp_folder.is_dir())
        copy_src = self.tmp_folder/'src'
        dst = self.tmp_folder/'dst'
        src_all = self.make_src_files_for_copy_tests(copy_src)

        for shift, fn in zip([0, 2], [copy_any, move_file]):
            # clean up
            self.make_folder_for_sure(dst)
            # test multiple source files
            fn(src_all[5+shift:7+shift], str(dst))
            self.assertTrue((dst/src_all[5+shift].name).is_dir())
            self.assertTrue((dst/src_all[6+shift].name).is_dir())
            fn(src_all[shift:2+shift], str(dst))
            for k in range(2):
                cur = k + shift
                fname = dst/('file%d.txt'%cur)
                f = fname.open()
                text = f.read()
                f.close()
                self.assertEqual('abcde%d'%cur, text)
            if move_file == fn:
                for k in [7,8,2,3]:
                    self.assertFalse(src_all[k].exists())

    def test_pandas_utility(self):
        # df_merge_update
        agg_df = pd.DataFrame({'idx': [1, 2, 3, 4, 5], 'data': [100, 101, 102, 103, 104]})
        agg_df = agg_df.set_index('idx')
        df = pd.DataFrame({'idx': [3, 4, 6, 7], 'data': [110, 111, 112, 113]})
        df = df.set_index('idx')

        agg_df = df_merge_update(agg_df, df)
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


if __name__ == '__main__':
    unittest.main()
