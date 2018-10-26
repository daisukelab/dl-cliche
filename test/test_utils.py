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
        ensure_folder(folder)
        self.assertTrue(folder.is_dir())

    def test_make_copy_to(self):
        self.assertTrue(self.tmp_folder.is_dir())

        copy_src = self.tmp_folder/'src'
        self.make_folder_for_sure(copy_src)
        for i in range(5):
            sub_folder = copy_src/('sub%d' % i)
            self.make_folder_for_sure(sub_folder)
        for i in range(5):
            sub_file = copy_src/('file%d.txt' % i)
            f = sub_file.open('w')
            f.write('abcde%d' % i)
            f.close()
        src_all = list(copy_src.iterdir())

        dst = self.tmp_folder/'dst'
        make_copy_to(dst, src_all, operation=symlink_file)
        for i in range(5):
            sub_folder = dst/('sub%d' % i)
            self.assertTrue(sub_folder.is_symlink())
        for i in range(5):
            sub_file = dst/('file%d.txt' % i)
            self.assertTrue(sub_file.is_symlink())
        ensure_delete(dst)

        make_copy_to(dst, src_all, operation=copy_any)
        for i in range(5):
            sub_folder = dst/('sub%d' % i)
            self.assertTrue(sub_folder.is_dir())
        for i in range(5):
            f = (dst/('file%d.txt' % i)).open()
            text = f.read()
            f.close()
            self.assertEqual('abcde%d' % i, text)

if __name__ == '__main__':
    unittest.main()
