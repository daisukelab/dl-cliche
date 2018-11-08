"""
## Instal MeCab + neologd

Thanks to https://qiita.com/ekzemplaro/items/c98c7f6698f130b55d53

```sh
$ sudo apt install mecab libmecab-dev mecab-ipadic-utf8
$ sudo apt install libmecab-dev
$ sudo apt install mecab-ipadic-utf8

$ git clone https://github.com/neologd/mecab-ipadic-neologd.git
$ cd mecab-ipadic-neologd && sudo bin/install-mecab-ipadic-neologd

$ pip install mecab-python3
($ pip install neologdn)
```
"""

from .nlp_base import TokenizerBase
import MeCab
import neologdn

class TokenizeByMeCab(TokenizerBase):
    def __init__(self, stop_words, normalize=True):
        super().__init__(stop_words, normalize)
        self.tagger = MeCab.Tagger("-Owakati")

    def tokenize(self, text):
        self.raw_tokens = self.tagger.parse(text)
        _tokens = [self._format(neologdn.normalize(w) if self.normalize else w)
                   for w in self.raw_tokens.split()]
        self.tokens = [w for w in _tokens if w is not '']
        return self.tokens
    
def get_mecab_tokenizer(stop_words=['\u3000'], normalize=True):
    return TokenizeByMeCab(stop_words=stop_words, normalize=normalize)
