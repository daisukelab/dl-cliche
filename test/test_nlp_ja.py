"""
NLP Ja test.

Prerequisite:
- This test expects dictionary: system_full.dic.
"""
import unittest
from dlcliche.utils import *
from dlcliche.test import *
from dlcliche.nlp_ja import *

class TestNlpJa(unittest.TestCase):
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

    def test_sudachi_tokenizer(self):
        tokenizer = get_sudachi_tokenizer()

        recursive_test_array(self, ['我が輩', 'は', '猫', 'だ'], tokenizer.tokenize('吾輩は猫である'))

if __name__ == '__main__':
    unittest.main()
