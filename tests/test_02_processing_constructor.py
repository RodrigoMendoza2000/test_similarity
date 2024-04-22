# ---------------------------------------------------------------
# Processing class testing
# Author: Rodrigo Alfredo Mendoza España
# Last modified: 21/04/2024
# ---------------------------------------------------------------

from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestProcessingConstructor(TestCase):

    # Test case for valid 'document_or_sentence' parameter
    def test_value_error_type(self):
        with self.assertRaises(ValueError):
            model = Processing(training_directory='./training_data',
                               test_directory='./test_data',
                               document_or_sentences='words',
                               lemmatize_or_stemming='lemmatize')

    # Test case for valid 'lemmatize_or_semming' parameter
    def test_lemmatize_processing(self):
        with self.assertRaises(ValueError):
            model = Processing(training_directory='./training_data',
                               test_directory='./test_data',
                               document_or_sentences='document',
                               lemmatize_or_stemming='strip')

    
