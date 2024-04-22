import pandas as pd


def get_auc(par_confusion_matrix: dict) -> float:
    """
    Calculate the Area Under the Curve (AUC) for the given confusion matrix.

    Args:
        par_confusion_matrix (dict): A dictionary representing the confusion matrix.

    Returns:
        float: The Area Under the Curve (AUC) calculated from the confusion matrix.
    """
    true_positive = par_confusion_matrix['true_positive']
    false_positive = par_confusion_matrix['false_positive']
    true_negative = par_confusion_matrix['true_negative']
    false_negative = par_confusion_matrix['false_negative']

    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)

    auc = (1 + tpr - fpr) / 2

    return auc


class Decision:
    """
        A class to perform decision making based on plagiarism analysis.

        Attributes:
            cosine_similarity_threshhold (float): Threshold for
                cosine similarity.
            plagiarism_percentage_threshhold (float): Threshold for
                        plagiarism percentage. A document will be considered
                        plagiarized if the percentage of plagiarized content
                        is greater than this threshold.
        """

    def __init__(self,
                 cosine_similarity_threshhold=0.7,
                 plagiarism_percentage_threshhold=0.35):
        self.cosine_similarity_threshhold = cosine_similarity_threshhold
        self.plagiarism_percentage_threshhold = \
            plagiarism_percentage_threshhold

    def get_plagiarism_sentences(self,
                                 processed_list: list,
                                 most_similar_documents: dict = {}) -> str:
        """
            Convert a processed list of sentences into a DataFrame.

            Args:
                processed_list (list): A list containing processed sentences.
                The list must contain tuples with the following format:
                    (sentence,
                    cosine_score,
                    file_name,
                    file_sentence_number,
                    similar_sentence)

            Returns:
                string: A DataFrame containing the processed sentences
                    along with their details.
            """

        display_text = ""
        df = pd.DataFrame(processed_list, columns=['sentence',
                                                   'cosine_score',
                                                   'file_name',
                                                   'file_sentence_number',
                                                   'similar_sentence'])

        # df = df.drop(df[df.cosine_score < self.cosine_similarity_threshhold]
        #              .index)

        if self.is_plagiarism(
                self.get_plagiarism_pct_sentences(processed_list)):
            display_text += "PLAGIARISM DETECTED\n\n"
        else:
            display_text += "PLAGIARISM NOT DETECTED\n\n"

        for index, row in df.iterrows():
            if row['cosine_score'] > self.cosine_similarity_threshhold:
                display_text += f"Plagiarized Sentence: " \
                                f"{row['sentence']} || " \
                                f"does not " \
                                f"present plagiarism\n\n "
            else:
                display_text += f"Sentence: '{row['sentence']}' || " \
                                f"presents plagiarism from  " \
                                f"'{row['file_name']}' sentence " \
                                f"'{row['similar_sentence']}'\n\n "

        display_text += f"Plagiarism percentage: " \
                        f"\n{self.get_plagiarism_pct_sentences(processed_list)}" \
                        f"\n\n "

        if most_similar_documents != {}:
            display_text += 'Most similar document(s): \n'
        for document, similarity in most_similar_documents.items():
            if similarity > self.cosine_similarity_threshhold:
                display_text += f"{document} " \
                                f"with similarity: {round(similarity, 2)}\n"

        return display_text

    def get_plagiarism_pct_sentences(self,
                                     processed_list: list
                                     ) -> int:
        """
        Calculate the percentage of plagiarism in a list of
            processed sentences.

        Args:
            processed_list (list): A list containing processed sentences.

        Returns:
            float: Percentage of plagiarism detected in the list.
                The formula for this percentage is:
                plagiarism_percentage = SUM((sentence_length * cosine_score) /
                    document_word_count)
        """
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

                sentence_length = len(original_sentence)
                if cosine_score > self.cosine_similarity_threshhold:
                    plagiarism_percentage += (sentence_length * cosine_score) \
                                             / paragraph_length

        return plagiarism_percentage

    def is_plagiarism(self, plagiarism_percentage: float) -> bool:
        """
        Check if the given plagiarism percentage exceeds the threshold.

        Args:
            plagiarism_percentage (float): Percentage of plagiarism.

        Returns:
            bool: True if plagiarism percentage exceeds the threshold,
                False otherwise.
        """
        return plagiarism_percentage > self.plagiarism_percentage_threshhold

    def get_confusion_matrix(self,
                             processed_files: dict,
                             original_files: dict) -> dict:
        """
        Generate a confusion matrix based on processed and original files.

        Args:
            processed_files (dict): A dictionary containing processed files.
                This processed files may be obtained from Processing
                class, get_training_results() method.
            original_files (dict): A dictionary containing original files.

        Returns:
            dict: A dictionary representing the confusion matrix.
        """
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for file_name, processed_list in processed_files.items():
            plagiarism_percentage = self.get_plagiarism_pct_sentences(
                processed_list
            )
            if original_files[file_name] is True:
                if self.is_plagiarism(plagiarism_percentage):
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if self.is_plagiarism(plagiarism_percentage):
                    false_positive += 1
                else:
                    true_negative += 1

        return {'true_positive': true_positive,
                'false_positive': false_positive,
                'true_negative': true_negative,
                'false_negative': false_negative}


if __name__ == "__main__":
    from processing import Processing

    doc2vec = Processing(training_directory='../training_data',
                         test_directory='../test_data',
                         document_or_sentences='sentences',
                         lemmatize_or_stemming='lemmatize')
    doc2vec.train_model()

    doc2vecdocuments = Processing(training_directory='../training_data',
                                  test_directory='../test_data',
                                  document_or_sentences='document',
                                  lemmatize_or_stemming='lemmatize')
    doc2vecdocuments.train_model()

    top = doc2vecdocuments.get_most_similar_documents(
        '../test_data/FID-02.txt')

    lst = doc2vec.get_most_similar_document_sentences(
        '../test_data/FID-02.txt')

    decision = Decision()
    print(decision.get_plagiarism_sentences(lst, top))
    print(decision.get_plagiarism_pct_sentences(lst))

    """training_dat = doc2vec.testing_data()

    dic = {
        'FID-01.txt': True,
        'FID-02.txt': True,
        'FID-03.txt': True,
        'FID-04.txt': True,
        'FID-05.txt': True,
        'FID-06.txt': True,
        'FID-07.txt': True,
        'FID-08.txt': True,
        'FID-09.txt': True,
        'FID-10.txt': True,
        'FID-11-mine.txt': False,
    }

    
    confusion_matrix = decision.get_confusion_matrix(training_dat, dic)

    print(get_auc(confusion_matrix))"""

    #
    # print(decision.get_plagiarism_pct_sentences(lst))
