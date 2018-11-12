from .nlp_base import TokenizerBase
import json
from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import config

class TokenizeBySudachi(TokenizerBase):
    def __init__(self, stop_words, normalize=False, mode=tokenizer.Tokenizer.SplitMode.B):
        super().__init__(stop_words, normalize)
        with open(config.SETTINGFILE, "r", encoding="utf-8") as f:
            settings = json.load(f)
        self.tokenizer = dictionary.Dictionary(settings).create()
        self.mode = mode
    def tokenize(self, text):
        self.raw_tokens = self.tokenizer.tokenize(self.mode, text.strip())
        _tokens = [self._format(w.normalized_form() if self.normalize else w.surface())
                   for w in self.raw_tokens]
        self.tokens = [w for w in _tokens if w is not '']
        return self.tokens

def get_sudachi_tokenizer(stop_words=['\u3000'], normalize=False):
    return TokenizeBySudachi(stop_words=stop_words, normalize=normalize)
