import os
import gensim
import smart_open
from sklearn.metrics import pairwise
from preprocessing import Preprocessing
import random
import warnings
warnings.filterwarnings("ignore")


def read_corpus(fname: str, tokens_only: bool=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            # print(line)
            tokens = gensim.utils.simple_preprocess(line)
            # tokens =
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


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

    def train_model(self, vector_size: int = 50, min_count: int = 2, epochs: int = 40):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
        self.model.build_vocab(self.train_corpus)
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_cosine_similarity_two_sentences(self, sentence1: str, sentence2: str):
        sentence1 = self.preprocessing.transform_prompt(sentence1)
        sentence2 = self.preprocessing.transform_prompt(sentence2)
        vector1 = self.model.infer_vector(sentence1)
        vector2 = self.model.infer_vector(sentence2)

        cosine_similarity = pairwise.cosine_similarity([vector1, vector2])
        return cosine_similarity

    def read_corpus(self, fname: str, tokens_only: bool = False):
        with smart_open.open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                # tokens = gensim.utils.simple_preprocess(line)
                tokens = self.preprocessing.transform_prompt(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

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

    def test_model(self):
        random_doc_id = random.randint(0, len(self.test_corpus) - 1)
        inferred_vector = self.model.infer_vector(self.test_corpus[random_doc_id])
        most_similar = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))

        print(f"Test Document ({random_doc_id}): «{' '.join(self.test_corpus[random_doc_id])}»\n")
        # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % self.model)
        for label, index in [('MOST', 0), ('MEDIAN', len(most_similar) // 2), ('LEAST', len(most_similar) - 1)]:
            # print(u'%s %s: «%s»\n' % (label, most_similar[index],
            # ' '.join(self.train_corpus[most_similar[index][0]].words)))
            print(f"{label} {most_similar[index]}")


if __name__ == '__main__':
    doc2vec = Doc2VecProcessing(training_directory='../training_data/all-training.tar',
                                test_directory='../test_data/test-data.tar')
    doc2vec.train_model()
    # print(doc2vec.get_cosine_similarity_two_sentences(['this', 'study', 'provided', 'data'], ['this', 'work', 'provides', 'data']))
    # print(doc2vec.get_cosine_similarity_two_sentences('this work provided data',
    #                                                   'this study provides data'))
    doc2vec.test_model()

"""
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[0])


import collections

counter = collections.Counter(ranks)
print(counter)

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


# Pick a random document from the corpus and infer a vector from the model
import random
doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))


# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))"""