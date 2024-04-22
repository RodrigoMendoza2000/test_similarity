# ---------------------------------------------------------------
# Decision class testing
# Author: Antonio Oviedo Paredes
# Last modified: 21/04/2024
# ---------------------------------------------------------------

from unittest import TestCase
from model.textpreprocessing import Preprocessing
from model.decision import Decision
from model.processing import Processing


class TestDecisionPlagiarisPercentage(TestCase):

    def setUp(self):
        self.model = Processing(training_directory='./training_data',
                                    test_directory='./test_data',
                                    document_or_sentences='sentences',
                                    lemmatize_or_stemming='lemmatize')
        self.model.train_model()
        
        self.d = Decision()

    # ------- Test cases to get percentage of plagiarism of a plagiarized text -------
    def test_plagiarism_1(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-01.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_2(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-02.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_3(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-03.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_4(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-04.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_5(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-05.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_6(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-06.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_7(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-07.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_8(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-08.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    def test_plagiarism_9(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/FID-09.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertTrue(
            self.d.is_plagiarism(pcg)
        )

    ## Document not detected correctly
    # def test_plagiarism_10(self):
    #     lst = self.model.get_most_similar_document_sentences('./test_data/FID-10.txt')
    #     pcg = self.d.get_plagiarism_pct_sentences(lst)
    #     self.assertTrue(
    #         self.d.is_plagiarism(pcg)
    #     )

    # ------- Test cases to get percentage of plagiarism of a non plagiarized text -------
    def test_no_plagiarism_1(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-001.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_2(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-002.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_5(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-005.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_12(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-012.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_20(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-020.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_26(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-026.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_40(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-040.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_47(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-047.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    ## Document not detected correctly
    # def test_no_plagiarism_56(self):
    #     lst = self.model.get_most_similar_document_sentences('./test_data/org-56.txt')
    #     pcg = self.d.get_plagiarism_pct_sentences(lst)
    #     self.assertFalse(
    #         self.d.is_plagiarism(pcg)
    #     )

    def test_no_plagiarism_59(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-059.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )
    
    def test_no_plagiarism_61(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-061.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_65(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-065.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_71(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-071.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_81(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-081.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_84(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-084.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_90(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-090.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_93(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-093.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_100(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-100.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )

    def test_no_plagiarism_110(self):
        lst = self.model.get_most_similar_document_sentences('./test_data/org-110.txt')
        pcg = self.d.get_plagiarism_pct_sentences(lst)
        self.assertFalse(
            self.d.is_plagiarism(pcg)
        )


    