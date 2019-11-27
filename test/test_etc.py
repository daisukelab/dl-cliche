"""
Log test.
"""
import unittest
from dlcliche.utils import *
from dlcliche.test import *
import re

class TestTest(unittest.TestCase):
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

    def test_create_dummy_file(self):
        filenames = []
        for fn in [None, 'aaaaaaaa', 't.t.t.t.bcd']:
            for suffix in [None, '.bar', '.longer']:
                for folder in [Path('/tmp'), Path('.')]:
                    for r in [True, False]:
                        dummy = create_dummy_file(1000, randomize=r, folder=folder, filename=fn, suffix=suffix)
                        self.assertTrue(dummy is not None)
                        self.assertTrue(os.path.getsize(dummy) == 1000) # f'{dummy} has size of {os.path.getsize(dummy)}.'
                        #print(f'Passed with {dummy} {"with random values" if r else "just empty"}.')
                        filenames.append(dummy)
        for f in filenames: ensure_delete(f)
        #print('OK')

    def test_tgz_all(self):
        test_base_folder = Path('/tmp/test_tgz')
        test_src_folder = test_base_folder/'src'
        zips = []
        try:
            child1, child2 = test_src_folder/'child1', test_src_folder/'child2'
            for d in [test_src_folder, child1, child2]:
                ensure_folder(d)
                for i in range(5):
                    create_dummy_file(size=100_000, randomize=True, folder=d, filename=f'foo{i}.tmp')
            #! ls -lR {test_src_folder}

            # zip child1
            files = [str(f).replace(str(child1.parent)+'/', '') for f in child1.iterdir()]
            for dest in [None, test_src_folder/'ch1.tgz']:
                generated = tgz_all(base_dir=test_src_folder, files=files, dest_tgz_path=dest)
                self.assertTrue(generated is not None)
                zips.append(generated)
            # ! ls -lR {test_src_folder}

            # zip child2
            files = [str(f).replace(str(child2.parent)+'/', '') for f in child2.iterdir()]
            for dest in [None, test_src_folder/'ch2.tgz']:
                generated = tgz_all(base_dir=test_src_folder, files=files, dest_tgz_path=dest)
                self.assertTrue(generated is not None)
                zips.append(generated)
            #! ls -lR {test_src_folder}

            # delete folders child2, test failure
            ensure_delete(child2)
            generated = tgz_all(base_dir=test_src_folder, files=files, dest_tgz_path=test_src_folder/'tobefail.tgz')
            self.assertTrue(generated is None)

            # zip entire src
            generated = tgz_all(base_dir=test_base_folder, files=['src'], dest_tgz_path=None)
            self.assertTrue(generated is not None)

            # no test
            generated = tgz_all(base_dir=test_base_folder, files=['src'], dest_tgz_path=test_base_folder/'entire.tgz', test=False)
            self.assertTrue(generated is not None)
            #!ls -lR {test_base_folder}

        finally:
            ensure_delete(test_base_folder)


if __name__ == '__main__':
    unittest.main()
