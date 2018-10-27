

import json
from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import config

class TokenizeBySudachi:
    def __init__(self, stop_words, normalize=True, mode=tokenizer.Tokenizer.SplitMode.B):
        with open(config.SETTINGFILE, "r", encoding="utf-8") as f:
            settings = json.load(f)
        self.tokenizer = dictionary.Dictionary(settings).create()
        self.stop_words = stop_words
        self.normalize = normalize
        self.mode = mode
    def _format(self, word):
        if word.isdigit():
            return '0'
        elif word in self.stop_words:
            return ''
        else:
            return word
    def tokenize(self, text):
        self.raw_tokens = self.tokenizer.tokenize(self.mode, text.strip())
        _tokens = [self._format(w.normalized_form() if self.normalize else w.surface())
                   for w in self.raw_tokens]
        self.tokens = [w for w in _tokens if w is not '']
        return self.tokens

def get_sudachi_tokenizer(stop_words=['\u3000'], normalize=True):
    return TokenizeBySudachi(stop_words=stop_words, normalize=normalize)
