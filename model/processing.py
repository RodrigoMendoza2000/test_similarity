# ---------------------------------------------------------------
# Processing class
# Authors: Rodrigo Alfredo Mendoza España
#          Antonio Oviedo Paredes
# Last modified: 21/04/2024
# ---------------------------------------------------------------

import os
import gensim
from sklearn.metrics import pairwise
from nltk.tokenize import sent_tokenize


class Processing:
    """
        Class for processing text data and training Doc2Vec
        models to detect plagiarism.

        Attributes:
            training_directory (str):
                The directory containing training documents. The directory
                must be outside the code directory.
            test_directory (str):
                The directory containing test documents. The directory
                must be outside the code directory
            document_or_sentences (str):
                Type of input data, either 'document' or 'sentences'.
            lemmatize_or_stemming (str):
                Preprocessing method, either 'lemmatize' or 'stemming'.
            preprocessing (Preprocessing):
                Instance of the Preprocessing class.
            model_dictionary (dict):
                Dictionary to store original documents and sentence numbers.
            train_corpus (list):
                List of TaggedDocuments for training the Doc2Vec model.
            test_corpus (list):
                List of tokens or TaggedDocuments for testing the model.
            model (gensim.models.doc2vec.Doc2Vec):
                Trained Doc2Vec model.

        Methods:
            train_model(vector_size=35, min_count=1, epochs=120):
                Train the Doc2Vec model.
            get_cosine_similarity_two_sentences(sentence1, sentence2):
                Calculate cosine similarity between two sentences.
            get_most_similar_objects(document_directory,
                                    threshhold=0.6, topn=3):
                Get most similar documents or sentences.
        """

    def __init__(self,
                 training_directory: str,
                 test_directory: str,
                 document_or_sentences: str = 'document',
                 lemmatize_or_stemming: str = 'lemmatize'):
        """
        Initialize Doc2VecProcessing with the given directories and settings.

        Args:
            training_directory (str):
                Directory containing training documents.
            test_directory (str):
                Directory containing test documents.
            document_or_sentences (str, optional):
                Type of input data, either 'document' or 'sentences'.
            lemmatize_or_stemming (str, optional):
                Preprocessing method, either 'lemmatize' or 'stemming'.
        """
        if document_or_sentences not in ['document', 'sentences']:
            raise ValueError(
                "Invalid value for document_or_sentences. "
                "Possible values: 'document', 'sentences'"
            )
        if lemmatize_or_stemming not in ['lemmatize', 'stemming']:
            raise ValueError(
                "Invalid value for lemmatize_or_stemming. "
                "Possible values: 'lemmatize', 'stemming'"
            )
        self.preprocessing = Preprocessing()
        self.lemmatize_or_stemming = lemmatize_or_stemming
        self.training_directory = training_directory
        self.test_directory = test_directory
        self.model_dictionary = {}
        self.document_or_sentences = document_or_sentences
        if self.document_or_sentences == 'document':
            self.train_corpus = list(self.__read_corpus(training_directory))
            self.test_corpus = list(self.__read_corpus(test_directory,
                                                       tokens_only=True))
        elif self.document_or_sentences == 'sentences':
            self.train_corpus = list(
                self.__read_corpus_sentences(training_directory)
            )
            self.test_corpus = list(self.__read_corpus_sentences(
                test_directory,
                tokens_only=True)
            )
        self.model = None

    def train_model(self,
                    vector_size: int = 35,
                    min_count: int = 1,
                    epochs: int = 120) -> None:
        """
        Train the Doc2Vec model with the specified parameters.

        Args:
            vector_size (int, optional):
                Dimensionality of the feature vectors. Defaults to 35.
            min_count (int, optional):
                Ignores all words with total frequency lower than this.
                Defaults to 1.
            epochs (int, optional):
                Number of iterations over the corpus. Defaults to 120.
        """
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size,
                                                   min_count=min_count,
                                                   epochs=epochs,
                                                   window=3,
                                                   dm=1)
        # Get all the unique words from all texts
        self.model.build_vocab(self.train_corpus)
        # Train the model with the specified parameters
        self.model.train(self.train_corpus,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)

    def get_cosine_similarity_two_sentences(self,
                                            sentence1: str,
                                            sentence2: str) -> float:
        """
        Calculate the cosine similarity between two sentences.

        Args:
            sentence1 (str): First sentence.
            sentence2 (str): Second sentence.

        Returns:
            float: Cosine similarity between the two sentences.
        """
        # Preprocess parameters for more accurate similarity
        sentence1 = self.preprocessing.transform_prompt(sentence1)
        sentence1_tokenized = self.preprocessing.tokenize(sentence1)
        sentence2 = self.preprocessing.transform_prompt(sentence2)
        sentence2_tokenized = self.preprocessing.tokenize(sentence2)
        vector1 = self.model.infer_vector(sentence1_tokenized)
        vector2 = self.model.infer_vector(sentence2_tokenized)

        cosine_similarity = pairwise.cosine_similarity([vector1, vector2])
        return cosine_similarity

    def __read_corpus(self,
                      directory: str,
                      tokens_only: bool = False
                      ) -> list:
        """
        Read documents from the specified directory for
        training or testing data.

        Args:
            directory (str):
                Path to the directory containing documents.
            tokens_only (bool, optional):
                Whether to return only tokens or TaggedDocuments.
                Defaults to False.

        Returns:
            list: List of TaggedDocuments if tokens_only is False,
                otherwise a list of document tokens.
        """
        document_tags = []
        for file_name in os.listdir(f'{directory}'):
            with open(f'{directory}/{file_name}', 'r',
                      encoding='ISO-8859-1') as file:
                if ".txt" in file_name:
                    text_preprocessed = self.preprocessing.transform_prompt(
                        " ".join(file.readlines()),
                        lemmatize_or_stemming=self.lemmatize_or_stemming
                    )
                    tokens = self.preprocessing.tokenize(text_preprocessed)
                    if tokens_only:
                        document_tags.append(tokens)
                    else:
                        document_tags.append(
                            gensim.models.doc2vec.TaggedDocument(tokens,
                                                                 [file_name])
                        )
        return document_tags

    def __read_corpus_sentences(self, directory, tokens_only: bool = False):
        """
        Read sentences from documents in the specified directory for
        training or testing data.

        Args:
            directory (str):
                Path to the directory containing documents.
            tokens_only (bool, optional):
                Whether to return only tokens or TaggedDocuments.
                Defaults to False.

        Returns:
            list: List of TaggedDocuments if tokens_only is False,
            otherwise a list of sentence tokens.
        """
        document_tags = []
        unique_id = 0
        for file_name in os.listdir(f'{directory}'):
            with open(f'{directory}/{file_name}',
                      'r',
                      encoding='ISO-8859-1') as file:
                if ".txt" in file_name:
                    line_number = 1
                    lines = " ".join(file.readlines())
                    sentences = sent_tokenize(lines)
                    for line in sentences:
                        sentence_preprocessed = \
                            self.preprocessing.transform_prompt(
                                line,
                                lemmatize_or_stemming=
                                self.lemmatize_or_stemming
                            )
                        tokens = \
                            self.preprocessing.tokenize(sentence_preprocessed)
                        if tokens_only:
                            document_tags.append(tokens)
                        else:
                            document_tags.append(
                                gensim.models.doc2vec.TaggedDocument(
                                    tokens,
                                    [unique_id]
                                )
                            )
                            # add to dictionary for later use
                            self.model_dictionary[unique_id] = [file_name,
                                                                line_number,
                                                                line]
                            line_number += 1
                            unique_id += 1
        return document_tags

    def get_most_similar_documents(self,
                                   document_directory,
                                   threshhold=0.6,
                                   topn=3) -> dict:
        """
        Get the most similar documents to the input document.

        Args:
            document_directory (str): Path to the input document.
            threshhold (float, optional): Minimum cosine similarity threshold.
                Defaults to 0.6.
            topn (int, optional): Number of most similar documents to return.
                Defaults to 3.

        Returns:
            dict: Dictionary containing the most similar documents and their
                cosine similarities. Keys are document indices and values are
                cosine similarity scores.
        """
        if self.document_or_sentences != 'document':
            raise Exception('To use this function the model must be set '
                            'to document')

        with open(document_directory, 'r', encoding='ISO-8859-1') as file:
            text_preprocessed = self.preprocessing.transform_prompt(
                " ".join(file.readlines()),
                lemmatize_or_stemming=self.lemmatize_or_stemming
            )
            tokens = self.preprocessing.tokenize(text_preprocessed)
        inferred_vector = self.model.infer_vector(tokens)
        most_similar = self.model.dv.most_similar([inferred_vector], topn=topn)

        dict_of_similar_documents = {}
        for sim in most_similar:
            if sim[1] > threshhold:
                dict_of_similar_documents[sim[0]] = sim[1]

        return dict_of_similar_documents

    def get_most_similar_document_sentences(self,
                                            document_directory) -> list:
        """
        Get the most similar sentences to each sentence in the input document.

        Args:
            document_directory (str): Path to the input document.

        Returns:
            list: List containing information about the most similar sentences.
                  Each element is a list with the following structure:
                  [original_sentence, cosine_similarity, file_name,
                  sentence_number, similar_sentence_text]
        """
        if self.document_or_sentences != 'sentences':
            raise Exception('To use this function the model must be set '
                            'to sentences')

        tokens_of_sentences = []
        with open(document_directory, 'r', encoding='ISO-8859-1') as file:
            lines = " ".join(file.readlines())
            sentences = sent_tokenize(lines)
            count = 0
            for sentence in sentences:
                count += 1
                # list containing the original sentence and
                # the altered sentence
                tokens_of_sentences.append([
                    sentence,
                    self.preprocessing.tokenize(
                        self.preprocessing.transform_prompt(
                            sentence,
                            lemmatize_or_stemming=self.lemmatize_or_stemming
                        ))
                ])

        most_similar_sentences = []
        for tokens in tokens_of_sentences:
            if tokens[1]:
                inferred_vector = self.model.infer_vector(tokens[1])
                most_similar = self.model.dv.most_similar([inferred_vector],
                                                          topn=1)
                most_similar_sentences.append([tokens[0],
                                               most_similar[0][1],
                                               self.model_dictionary[
                                                   most_similar[0][0]][0],
                                               self.model_dictionary[
                                                   most_similar[0][0]][1],
                                               self.model_dictionary[
                                                   most_similar[0][0]][2]])

        return most_similar_sentences

    def get_training_results_sentences(self) -> dict:
        testing_results = {}
        for file in os.listdir(self.test_directory):
            testing_results[file] = self.get_most_similar_document_sentences(
                f"{self.test_directory}/{file}"
            )

        return testing_results

    def get_training_results_documents(self) -> dict:
        testing_results = {}
        for file in os.listdir(self.test_directory):
            testing_results[file] = self.get_most_similar_documents(
                f"{self.test_directory}/{file}",
                threshhold=0.6,
            )

        return testing_results


if __name__ == '__main__':
    from textpreprocessing import Preprocessing

    doc2vec = Processing(training_directory='../training_data',
                         test_directory='../test_data',
                         document_or_sentences='document',
                         lemmatize_or_stemming='lemmatize')

    doc2vec.train_model()

    # lemmatize presents way better results)
    # doc2vec.train_model()
    # print(doc2vec.get_most_similar_documents('../test_data/org-090.txt'))
    # doc2vec.testing_data()
    # lista = doc2vec.get_most_similar_objects('../test_data/FID-11-mine.txt')
else:
    from model.textpreprocessing import Preprocessing
