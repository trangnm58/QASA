import re
import spacy as sp
from spacy.symbols import ORTH, LEMMA
from nltk.corpus import stopwords

from utils import Timer
from constants import URL


class Spacy:
    nlp = None

    @staticmethod
    def load_spacy():
        t = Timer()
        t.start('Load SpaCy', verbal=True)
        Spacy.nlp = sp.load('en_core_web_lg')
        t.stop()
        Spacy.nlp.tokenizer.add_special_case("+/-", [{ORTH: "+/-", LEMMA: "+/-"}])

    @staticmethod
    def parse(text):
        if Spacy.nlp is None:
            Spacy.load_spacy()

        return Spacy.nlp(text)


class SpacyTokenizer:
    @staticmethod
    def tokenize(sent):
        tokens = Spacy.parse(sent)
        return [t.string.strip() for t in tokens]


class UrlNormalizer:
    def __init__(self, replace=URL):
        self.replace = replace

    def process(self, text):
        text = re.sub(r"<a.*</a>", self.replace, text)
        return text


class PreProcess:
    def __init__(self):
        self.tokenizer = SpacyTokenizer()
        self.url_normalizer = UrlNormalizer()
        self.simple_tokenizer = lambda doc: re.split("[ _]", doc)
        self.stop_words = set(stopwords.words("english"))

    def process(self, text, remove_stop_words=True, simple_tokenize=False, url_norm=False):
        """
        Normalize then tokenize the text, remove stop words
        :string text: a sentence or a paragraph
        :return: list of tokens
        """
        if url_norm:
            text = self.url_normalizer.process(text)

        cleaned_text = self._clean(text)

        if simple_tokenize:
            tokens = self.simple_tokenizer(cleaned_text)
        else:
            tokens = self.tokenizer.tokenize(cleaned_text)

        if remove_stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    @staticmethod
    def _clean(text):
        """
        Remove all characters that are not word nor number characters.
        :return: cleaned text
        """
        cleaned = re.sub("[^\w\d]", " ", text).lower()
        cleaned = re.sub(" +", " ", cleaned)

        cleaned = cleaned.strip()
        return cleaned
