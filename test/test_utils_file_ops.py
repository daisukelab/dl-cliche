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

    def make_folder_for_sure(self, folder):
        ensure_delete(folder)
        ensure_folder(folder)
        self.assertTrue(folder.is_dir())

    def make_files_in_folder(self, folder, K=5, content_prefix='abcde'):
        folder = Path(folder)
        for i in range(K):
            sub_file = folder/('file%d.txt' % i)
            f = sub_file.open('w')
            f.write(f'{content_prefix}{i}')
            f.close()

    def make_src_files_for_copy_tests(self, src_folder):
        self.make_folder_for_sure(src_folder)
        for i in range(5):
            sub_folder = src_folder/('sub%d' % i)
            self.make_folder_for_sure(sub_folder)
            self.make_files_in_folder(sub_folder)
        self.make_files_in_folder(src_folder)
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

    def test_4_expand_path(self):
        self.assertTrue(self.tmp_folder.is_dir())
        copy_src = self.tmp_folder/'src'
        src_all = self.make_src_files_for_copy_tests(copy_src)
        files = [f for f in src_all if f.is_file()]
        # sub case 1: all files there
        result_set = set(expand_path(copy_src/'*.txt'))
        self.assertTrue(result_set == set(files))
        # sub case 2: single name match
        result = expand_path(copy_src/'sub0')
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0]), str(copy_src/'sub0'))
        # sub case 3: no match
        result = expand_path(copy_src/'nothing')
        self.assertEqual(len(result), 0)

    def test_5_copy_with_prefix(self):
        self.assertTrue(self.tmp_folder.is_dir())
        copy_src = self.tmp_folder/'src'
        dst = self.tmp_folder/'dst'
        src_all = self.make_src_files_for_copy_tests(copy_src)
        self.make_folder_for_sure(dst)
        # sub case 1: all files in src_all
        files = [f for f in src_all if f.is_file()]
        copy_with_prefix(files, dst, 'Root', symlinks=False)
        for f in files:
            expected_f = dst/('Root'+f.name)
            self.assertTrue(expected_f.is_file())
            ensure_delete(expected_f)
        copy_with_prefix(copy_src/'*.txt', dst, 'Root', symlinks=False)
        for f in files:
            expected_f = dst/('Root'+f.name)
            self.assertTrue(expected_f.is_file())
            ensure_delete(expected_f)
        # sub case 2: all files as symlink
        copy_with_prefix(files, dst, 'Root', symlinks=True)
        for f in files:
            expected_f = dst/('Root'+f.name)
            self.assertTrue(expected_f.is_file() and expected_f.is_symlink())
            ensure_delete(expected_f)
        copy_with_prefix(copy_src/'*.txt', dst, 'Root', symlinks=True)
        for f in files:
            expected_f = dst/('Root'+f.name)
            self.assertTrue(expected_f.is_file() and expected_f.is_symlink())
            ensure_delete(expected_f)
        # sub case 3: raise exception if dest is not a folder
        try:
            copy_with_prefix(files[0], files[1], 'dummy', symlinks=False)
            result = False
        except:
            result = True
        self.assertTrue(result)
        # sub case 4: raise exception if any of input files are not a file
        try:
            copy_with_prefix(src_all, dst, 'dummy')
            result = False
        except:
            result = True
        self.assertTrue(result)
        try:
            copy_with_prefix(copy_src/'*', dst, 'dummy')
            result = False
        except:
            result = True
        self.assertTrue(result)
        # sub case 5: complex copy ['sub[1-3].txt', 'sub1/*.txt']
        copy_with_prefix(copy_src/'sub0/*.txt', copy_src/'sub0', '0_')
        copy_with_prefix(copy_src/'sub1/*.txt', copy_src/'sub1', '1_')
        copy_with_prefix([copy_src/'sub0/0*.txt', copy_src/'sub1/1*.txt'], dst, 'dst_')
        self.assertEqual(len(files) * 2, len(expand_path(dst/'dst_*.txt')))
        

if __name__ == '__main__':
    unittest.main()
