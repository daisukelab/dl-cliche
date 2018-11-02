"""
Log test.
"""
import unittest
from dlcliche.utils import *
from dlcliche.test import *
import re

class TesLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        ensure_delete('logtest')
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_1_log(self):
        global log
        log = get_logger()
        log.info('logging info')
        log.debug('logging debug')
        log.error('logging error')
        # you just have to see them...

    def test_2_log(self):
        # supposed to run after 1_log
        global log
        log.info('you will see this if global log is accessible.')

    def test_3_file_log(self):
        log = get_logger('logtest', level=logging.INFO, print=False, output_file='logtest/foo/bar/test.txt')
        log.info('logging info')
        log.debug('logging debug')
        log.error('logging error')
        with open('logtest/foo/bar/test.txt') as f:
            contents = f.read()
        self.assertEqual(re.sub('[0-9]', '0', contents),
            '0000-00-00 00:00:00,000 logtest test_0_file_log [INFO]: logging info\n'\
            '0000-00-00 00:00:00,000 logtest test_0_file_log [ERROR]: logging error\n')

    def test_4_file_log(self):
        log = get_logger('logtest', level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s',
                         print=False, output_file='logtest/foo/bar/test2.txt')
        log.info('logging info')
        with open('logtest/foo/bar/test2.txt') as f:
            contents = f.read()
        self.assertEqual(re.sub('[0-9]', '0', contents),
            '0000-00-00 00:00:00,000: INFO: logging info\n')

if __name__ == '__main__':
    unittest.main()
