from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestProcessingTraining(TestCase):

    def setUp(self):
        self.model = Processing(training_directory='./training_data',
                           test_directory='./test_data',
                           document_or_sentences='sentences',
                           lemmatize_or_stemming='lemmatize')


    # Test case for valid 'vector_size' parameter
    def test_value_error_size_neg(self):
        with self.assertRaises(ValueError):
            self.model.train_model(vector_size=-5,
                                   min_count=1,
                                   epochs=120)

    # Test case for valid 'epochs' parameter
    def test_value_error_epochs_neg(self):
        with self.assertRaises(ValueError):
            self.model.train_model(vector_size=35,
                                   min_count=1,
                                   epochs=-5)
            
    # Test case for valid 'epochs' parameter
    def test_value_error_epochs_0(self):
        with self.assertRaises(ValueError):
            self.model.train_model(vector_size=35,
                                   min_count=1,
                                   epochs=0)
            
    
