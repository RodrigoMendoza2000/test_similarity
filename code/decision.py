class Decision:
    def __init__(self):
        pass

    # TODO: plagiarism percentage =
    #  SUM((number of words in sentence * cosine similarity if cosine similarity > X) / total words in text)
    # X = parameter

    def get_plagiarism_pct_sentences(self,
                                     processed_list: list,
                                     cosine_similarity_threshhold: int = 0.7
    ) -> int:
        paragraph_length = 0
        plagiarism_percentage = 0
        for sentence in processed_list:
            if sentence is not None:
                original_sentence = sentence[0]
                paragraph_length += len(original_sentence)

        for sentence in processed_list:

            if sentence is not None:
                original_sentence = sentence[0]
                cosine_score = sentence[1]
                # file_name = sentence[2]
                # similar_sentence = sentence[4]

                sentence_length = len(original_sentence)
                print(cosine_score)
                if cosine_score > cosine_similarity_threshhold:
                    plagiarism_percentage += (sentence_length * cosine_score) \
                                             / paragraph_length

        return plagiarism_percentage


if __name__ == "__main__":
    from processing import Processing

    doc2vec = Processing(training_directory='../training_data',
                         test_directory='../test_data',
                         document_or_sentences='sentences',
                         lemmatize_or_stemming='lemmatize')
    doc2vec.train_model()

    lst = doc2vec.get_most_similar_document_sentences('../test_data/FID-09.txt')
    decision = Decision()
    print(decision.get_plagiarism_pct_sentences(lst))
