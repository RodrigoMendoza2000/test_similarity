from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestPreprocessingPhases(TestCase):

    # Test case for replacing special words
    def test_replace_words(self):
        p = Preprocessing()
        p.prompt = '(ai) is a an if or a01752114@tec.mx'
        p._Preprocessing__replace_words()
        self.assertEqual(
            '(artificial intelligence) is a an if or a01752114@tec.mx',
            p.prompt
        )

    # Test case for removing stop words
    def test_remove_stop_words(self):
        p = Preprocessing()
        p.prompt = '(ai) is a an if or a01752114@tec.mx'
        p._Preprocessing__remove_stopwords()
        self.assertEqual(
            'ai a01752114 tec mx',
            p.prompt
        )