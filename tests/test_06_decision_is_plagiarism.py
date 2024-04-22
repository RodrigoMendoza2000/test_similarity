# ---------------------------------------------------------------
# Decision class testing
# Author: Diego Yunoe Sierra Díaz
# Last modified: 21/04/2024
# ---------------------------------------------------------------

from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestDecisionPlagiarismCheck(TestCase):

    def setUp(self):
        self.d = Decision(cosine_similarity_threshhold=0.7,
                          plagiarism_percentage_threshhold=0.35)
    
    # Test case for plagiarism percentage < 'plagiarism_percentage_threshhold'
    def test_is_plagiarism_less(self):
        self.assertFalse(self.d.is_plagiarism(0.10))

    # Test case for plagiarism percentage == 'plagiarism_percentage_threshhold'
    def test_is_plagiarism_equal(self):
        self.assertFalse(self.d.is_plagiarism(0.35))

    # Test case for plagiarism percentage > 'plagiarism_percentage_threshhold'
    def test_is_plagiarism_greater(self):
        self.assertTrue(self.d.is_plagiarism(0.40))