import os
import gensim
import smart_open
from sklearn.metrics import pairwise
from preprocessing import Preprocessing
import random


class Doc2VecProcessing:
    def __init__(self, training_directory: str, test_directory: str):
        self.preprocessing = Preprocessing()
        self.training_directory = training_directory
        self.test_directory = test_directory
        self.train_corpus = list(self.read_corpus_directly('training_data'))
        self.test_corpus = list(self.read_corpus_directly('test_data', tokens_only=True))
        # self.train_corpus = list(self.read_corpus(self.training_directory))
        # self.test_corpus = list(self.read_corpus(self.test_directory, tokens_only=True))
        self.model = None

    def train_model(self, vector_size: int = 50, min_count: int = 2, epochs: int = 80):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
        # Get all the unique words from all texts
        self.model.build_vocab(self.train_corpus)
        # Train the model with the specified parameters
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_cosine_similarity_two_sentences(self, sentence1: str, sentence2: str):
        # Preprocess parameters for more accurate similarity
        sentence1 = self.preprocessing.transform_prompt(sentence1)
        sentence2 = self.preprocessing.transform_prompt(sentence2)
        vector1 = self.model.infer_vector(sentence1)
        vector2 = self.model.infer_vector(sentence2)

        cosine_similarity = pairwise.cosine_similarity([vector1, vector2])
        return cosine_similarity

    # Read all the documents for training or testing data.
    # In case of training data return a list of TaggedDocuments for the model
    # In case of testing data return a list of tokens of the document
    def read_corpus(self, fname: str, tokens_only: bool = False):
        with smart_open.open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                tokens = self.preprocessing.transform_prompt(line)
                if tokens_only:
                    yield tokens
                else:
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    # Do the same as read_corpus but without using smart_open. Way more reliable
    def read_corpus_directly(self, directory, tokens_only: bool = False):
        document_tags = []
        for file_name in os.listdir(f'../{directory}'):
            with open(f'../{directory}/{file_name}', 'r', encoding='ISO-8859-1') as file:
                if ".txt" in file_name:
                    tokens = self.preprocessing.transform_prompt(" ".join(file.readlines()))
                    if tokens_only:
                        document_tags.append(tokens)
                    else:
                        document_tags.append(gensim.models.doc2vec.TaggedDocument(tokens, [file_name]))
        return document_tags

    # Pick a random test file from the given tests
    def test_model_single(self):
        random_doc_id = random.randint(0, len(self.test_corpus) - 1)
        inferred_vector = self.model.infer_vector(self.test_corpus[random_doc_id])
        most_similar = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))

        print(f"Test Document ({random_doc_id}): «{' '.join(self.test_corpus[random_doc_id])}»\n")
        # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % self.model)
        for label, index in [('MOST', 0), ('MEDIAN', len(most_similar) // 2), ('LEAST', len(most_similar) - 1)]:
            # print(u'%s %s: «%s»\n' % (label, most_similar[index],
            # ' '.join(self.train_corpus[most_similar[index][0]].words)))
            print(f"{label} {most_similar[index]}")

    #
    def get_most_similar_documents(self, document_directory, threshhold=0.6):
        with open(document_directory, 'r', encoding='ISO-8859-1') as file:
            tokens = self.preprocessing.transform_prompt(" ".join(file.readlines()))
        print(tokens)
        inferred_vector = self.model.infer_vector(tokens)
        most_similar = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))

        list_of_similar_documents = []
        for sim in most_similar:
            if sim[1] > threshhold:
                list_of_similar_documents.append(sim)

        return sim


if __name__ == '__main__':
    doc2vec = Doc2VecProcessing(training_directory='../training_data/all-training.tar',
                                test_directory='../test_data/test-data.tar')
    doc2vec.train_model()
    # print(doc2vec.get_cosine_similarity_two_sentences(['this', 'study', 'provided', 'data'],
    # ['this', 'work', 'provides', 'data']))
    # print(doc2vec.get_cosine_similarity_two_sentences('this work provided data',
    #                                                   'this study provides data'))
    # doc2vec.test_model()
    # print(doc2vec.get_score(['Artificial Intelligence is smart']))
    doc2vec.get_most_similar_documents('../test_data/FID-01-02.txt')