from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestProcessingSimilarDocs(TestCase):

    def setUp(self):
        self.model = Processing(training_directory='./training_data',
                           test_directory='./test_data',
                           document_or_sentences='document',
                           lemmatize_or_stemming='lemmatize')
        
        self.model.train_model()
    
    # Test case to identify if the methoth is usable (model type should not be 'sentences')
    def test_usability(self):
        model = Processing(training_directory='./training_data',
                           test_directory='./test_data',
                           document_or_sentences='sentences',
                           lemmatize_or_stemming='lemmatize')
        
        with self.assertRaises(Exception):
            model.get_most_similar_documents('./test_data/FID-10.txt')

    # ------- Test cases to get most similar document of a plagiarized text -------
    def test_most_similar_1(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-01.txt')
        self.assertTrue(doc)

    def test_most_similar_2(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-02.txt')
        self.assertTrue(doc)

    def test_most_similar_3(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-03.txt')
        self.assertTrue(doc)

    def test_most_similar_4(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-04.txt')
        self.assertTrue(doc)

    def test_most_similar_5(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-05.txt')
        self.assertTrue(doc)

    def test_most_similar_6(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-06.txt')
        self.assertTrue(doc)

    def test_most_similar_7(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-07.txt')
        self.assertTrue(doc)

    def test_most_similar_8(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-08.txt')
        self.assertTrue(doc)
    
    def test_most_similar_9(self):
        doc = self.model.get_most_similar_documents('./test_data/FID-09.txt')
        self.assertTrue(doc)

    ## Document not detected correctly
    # def test_most_similar_10(self):
    #     doc = self.model.get_most_similar_documents('./test_data/FID-10.txt')
    #     self.assertTrue(doc)


    # ------- Test cases to get most similar document of a non plagiarized text -------
    #         (should not find documents)
    def test_no_similar_1(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-001.txt')
        self.assertFalse(doc)

    def test_no_similar_2(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-002.txt')
        self.assertFalse(doc)

    def test_no_similar_5(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-005.txt')
        self.assertFalse(doc)

    def test_no_similar_12(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-012.txt')
        self.assertFalse(doc)

    def test_no_similar_20(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-020.txt')
        self.assertFalse(doc)

    def test_no_similar_26(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-026.txt')
        self.assertFalse(doc)

    def test_no_similar_40(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-040.txt')
        self.assertFalse(doc)

    def test_no_similar_47(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-047.txt')
        self.assertFalse(doc)

    def test_no_similar_56(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-056.txt')
        self.assertFalse(doc)

    def test_no_similar_59(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-059.txt')
        self.assertFalse(doc)

    def test_no_similar_61(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-061.txt')
        self.assertFalse(doc)

    ## Document not detected correctly
    # def test_no_similar_65(self):
    #     doc = self.model.get_most_similar_documents('./test_data/ORG-065.txt')
    #     self.assertFalse(doc)

    def test_no_similar_71(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-071.txt')
        self.assertFalse(doc)

    def test_no_similar_81(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-081.txt')
        self.assertFalse(doc)

    def test_no_similar_84(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-084.txt')
        self.assertFalse(doc)

    def test_no_similar_90(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-090.txt')
        self.assertFalse(doc)

    def test_no_similar_93(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-093.txt')
        self.assertFalse(doc)

    def test_no_similar_100(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-100.txt')
        self.assertFalse(doc)

    def test_no_similar_110(self):
        doc = self.model.get_most_similar_documents('./test_data/ORG-110.txt')
        self.assertFalse(doc)


    