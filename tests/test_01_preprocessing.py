# ---------------------------------------------------------------
# Preprocessing class testing
# Author: Diego Yunoe Sierra Díaz
# Last modified: 21/04/2024
# ---------------------------------------------------------------

from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestPreprocessing(TestCase):

    def setUp(self):
        self.p = Preprocessing()

    # Test case for valid 'lemmatize_or_stemming' parameter
    def test_value_error(self):
        with self.assertRaises(ValueError):
            self.p.transform_prompt("Artificial Intellignece",
                                    lemmatize_or_stemming="strip")

    # Test case for lemmatize preprocessing
    def test_lemmatize_processing(self):
        with open(f'./test_data/org-001.txt', 'r') as file:
            text = " ".join(file.readlines())
            transformed = self.p.transform_prompt(text,
                                                  lemmatize_or_stemming='lemmatize')

            self.assertEqual("study provided content analysis study aiming disclose artificial intelligence artificial intelligence applied education sector explore potential research trend challenge artificial intelligence education total 100 paper including 63 empirical paper 74 study 37 analytic paper selected education educational research category social science citation index database 2010 2020 content analysis showed research question could classified development layer classification matching recommendation deep learning application layer feedback reasoning adaptive learning integration layer affection computing role playing immersive learning gamification moreover four research trend including internet thing swarm intelligence deep learning neuroscience well assessment artificial intelligence education suggested investigation however also proposed challenge education may caused artificial intelligence regard inappropriate use artificial intelligence technique changing role teacher student well social ethical issue result provide insight overview artificial intelligence used education domain help strengthen theoretical foundation artificial intelligence education provides promising channel educator artificial intelligence engineer carry collaborative research",
                             transformed)

    # Test case for stem preprocessing
    def test_stem_processing(self):
        with open(f'./test_data/org-001.txt', 'r') as file:
            text = " ".join(file.readlines())
            transformed = self.p.transform_prompt(text,
                                                  lemmatize_or_stemming='stem')

            self.assertEqual("study provided content analysis studies aiming disclose artificial intelligence artificial intelligence applied education sector explore potential research trends challenges artificial intelligence education total 100 papers including 63 empirical papers 74 studies 37 analytic papers selected education educational research category social sciences citation index database 2010 2020 content analysis showed research questions could classified development layer classification matching recommendation deep learning application layer feedback reasoning adaptive learning integration layer affection computing role playing immersive learning gamification moreover four research trends including internet things swarm intelligence deep learning neuroscience well assessment artificial intelligence education suggested investigation however also proposed challenges education may caused artificial intelligence regard inappropriate use artificial intelligence techniques changing roles teachers students well social ethical issues results provide insights overview artificial intelligence used education domain helps strengthen theoretical foundation artificial intelligence education provides promising channel educators artificial intelligence engineers carry collaborative research",
                             transformed)
    
    # Test case for tokenizing text
    def test_tokenize(self):
        self.assertEqual(['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.'],
                         self.p.tokenize('''Good muffins cost $3.88\nin New York.  Please buy me two of them.\n\nThanks.'''))
        
    # Test case for 