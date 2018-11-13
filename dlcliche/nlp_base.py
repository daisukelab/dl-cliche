import neologdn

class TokenizerBase:
    def __init__(self, stop_words, normalize):
        self.stop_words = stop_words
        self.normalize = normalize
    def _basic_normalize(self, word):
        if not self.normalize: return word
        word = str(word).lower()
        word = neologdn.normalize(word)
        return word
    def _format(self, word):
        word = self._basic_normalize(word)
        if word.isdigit():
            return '0'
        elif word in self.stop_words:
            return ''
        else:
            return word
