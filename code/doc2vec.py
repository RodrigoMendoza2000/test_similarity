import os
import gensim
from sklearn.metrics import pairwise
from preprocessing import Preprocessing
from nltk.tokenize import sent_tokenize
import random


class Doc2VecProcessing:
    def __init__(self, training_directory: str, test_directory: str, document_or_sentences='document', lemmatize_or_stemming='lemmatize'):
        self.preprocessing = Preprocessing()
        self.lemmatize_or_stemming = lemmatize_or_stemming
        self.training_directory = training_directory
        self.test_directory = test_directory
        # Dictionary to store the original document and the sentence number to retrieve them later
        self.model_dictionary = {}
        self.document_or_sentences = document_or_sentences
        if self.document_or_sentences == 'document':
            self.train_corpus = list(self.__read_corpus(training_directory))
            self.test_corpus = list(self.__read_corpus(test_directory, tokens_only=True))
        elif self.document_or_sentences == 'sentences':
            self.train_corpus = list(self.__read_corpus_sentences(training_directory))
            self.test_corpus = list(self.__read_corpus_sentences(test_directory, tokens_only=True))
        self.model = None

    def train_model(self, vector_size: int = 25, min_count: int = 2, epochs: int = 130):
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
    def __read_corpus(self, directory, tokens_only: bool = False):
        document_tags = []
        for file_name in os.listdir(f'../{directory}'):
            with open(f'../{directory}/{file_name}', 'r', encoding='ISO-8859-1') as file:
                if ".txt" in file_name:
                    tokens = self.preprocessing.transform_prompt(" ".join(file.readlines()), lemmatize_or_stemming=self.lemmatize_or_stemming)
                    if tokens_only:
                        document_tags.append(tokens)
                    else:
                        document_tags.append(gensim.models.doc2vec.TaggedDocument(tokens, [file_name]))
        return document_tags

    def __read_corpus_sentences(self, directory, tokens_only: bool = False):
        document_tags = []
        unique_id = 0
        for file_name in os.listdir(f'../{directory}'):
            with open(f'../{directory}/{file_name}', 'r', encoding='ISO-8859-1') as file:
                if ".txt" in file_name:
                    line_number = 1
                    lines = " ".join(file.readlines())
                    sentences = sent_tokenize(lines)
                    for line in sentences:
                        tokens = self.preprocessing.transform_prompt(line, lemmatize_or_stemming=self.lemmatize_or_stemming)
                        if tokens_only:
                            document_tags.append(tokens)
                        else:
                            document_tags.append(gensim.models.doc2vec.TaggedDocument(tokens, [unique_id]))
                            # add to dictionary for later use
                            self.model_dictionary[unique_id] = [file_name, line_number, line]
                            line_number += 1
                            unique_id += 1
        return document_tags

    # Pick a random test file from the given tests
    def __test_model_single(self):
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
    def __get_most_similar_documents(self, document_directory, threshhold=0.6):
        with open(document_directory, 'r', encoding='ISO-8859-1') as file:
            tokens = self.preprocessing.transform_prompt(" ".join(file.readlines()), lemmatize_or_stemming=self.lemmatize_or_stemming)
        inferred_vector = self.model.infer_vector(tokens)
        most_similar = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))

        list_of_similar_documents = []
        for sim in most_similar:
            if sim[1] > threshhold:
                list_of_similar_documents.append(sim)

        return list_of_similar_documents

    def __get_most_similar_document_sentences(self, document_directory):
        tokens_of_sentences = []
        with open(document_directory, 'r', encoding='ISO-8859-1') as file:
            lines = " ".join(file.readlines())
            sentences = sent_tokenize(lines)
            count = 0
            for sentence in sentences:
                count += 1
                # list containing the original sentence and the altered sentence
                tokens_of_sentences.append([sentence, self.preprocessing.transform_prompt(sentence, lemmatize_or_stemming=self.lemmatize_or_stemming)])

        # list to store [sentence, most_similar_sentence_cosine_similarity, most_similar_sentence_file_name,
        # most_similar_sentence_line_number, most_similar_sentence_text]
        most_similar_sentences = [[]]
        for tokens in tokens_of_sentences:
            if tokens[1]:
                inferred_vector = self.model.infer_vector(tokens[1])
                most_similar = self.model.dv.most_similar([inferred_vector], topn=1)
                most_similar_sentences.append([tokens[0],
                                               most_similar[0][1],
                                               self.model_dictionary[most_similar[0][0]][0],
                                               self.model_dictionary[most_similar[0][0]][1],
                                               self.model_dictionary[most_similar[0][0]][2]])

        return most_similar_sentences

    # Cosine similarity
    def get_most_similar_objects(self, document_directory):
        if self.document_or_sentences == 'document':
            return self.__get_most_similar_documents(document_directory)
        elif self.document_or_sentences == 'sentences':
            return self.__get_most_similar_document_sentences(document_directory)


if __name__ == '__main__':
    doc2vec = Doc2VecProcessing(training_directory='training_data',
                                test_directory='test_data',
                                document_or_sentences='document',
                                lemmatize_or_stemming='lemmatize')
                                # lemmatize presents way better results)
    doc2vec.train_model()
    # print(doc2vec.get_cosine_similarity_two_sentences(['this', 'study', 'provided', 'data'],
    # ['this', 'work', 'provides', 'data']))
    # print(doc2vec.get_cosine_similarity_two_sentences('this work provided data',
    #                                                   'this study provides data'))
    # doc2vec.test_model()
    # print(doc2vec.get_score(['Artificial Intelligence is smart']))
    print(doc2vec.get_most_similar_objects('../test_data/FID-10.txt'))
    # lista = doc2vec.get_most_similar_objects('../test_data/FID-11-mine.txt')
    # for l in lista:
    #     print(l)
    # TODO: aSK THE TEACHER IF ITS ONLY FOR A WHOLE DOCUMENT OR SENTENCE BY SENTENCE AND EVALUATE WITH DOCUMENTS BECAUSE IM GETTING POOR RESULTS PARAPHRASING THE SENTENCES
    # GETTING GOOD RESULTS WITH DOCUMENT SIMILARITY
    # TODO: MAYBE IMPLEMENT A NEW MODEL FOR DOCUMENTS WHERE IT IS TRAINGED ONLY BY THE SENTENCES IN THE MOST SIMILAR DOCUMENTS???? I DONT THINK THERE IS ENOUGH DATA