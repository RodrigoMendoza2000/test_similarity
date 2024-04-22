from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestProcessingCosine(TestCase):

    def setUp(self):
        self.model = Processing(training_directory='./training_data',
                           test_directory='./test_data',
                           document_or_sentences='sentences',
                           lemmatize_or_stemming='lemmatize')
        
        self.model.train_model()


    # Test case very similar sentences
    def test_cosine_similarity_1(self):
        s1 = 'AI algorithms process vast amounts of data to make predictions, while machine learning models learn from data to make forecasts.'
        s2 = 'AI methods process vast amounts of data to make predictions, while machine learning models learn from data to make forecasts.'

        cosine = self.model.get_cosine_similarity_two_sentences(s1, s2)

        self.assertGreater(cosine[0][1], 0.8)

    # Test case for very different sentences
    def test_cosine_similarity_0(self):
        s1 = 'Deep learning is transforming medicine by accurately detecting diseases early from medical images.'
        s2 = 'AI in energy management is optimizing electricity usage in smart cities, cutting costs and reducing environmental impact.'

        cosine = self.model.get_cosine_similarity_two_sentences(s1, s2)

        self.assertLess(cosine[0][1], 0.2)
            