# ---------------------------------------------------------------
# Main test file
# Author: Rodrigo Alfredo Mendoza España
# Last modified: 21/04/2024
# ---------------------------------------------------------------
import os

import pandas as pd

from model.processing import Processing
from model.decision import Decision, get_auc, generate_pdf

def plagiarism_types_report():
    doc2vec_documents = Processing(training_directory='./training_data',
                                   test_directory='./test_data',
                                   document_or_sentences='document',
                                   lemmatize_or_stemming='lemmatize')
    doc2vec_documents.train_model()

    lst = doc2vec_documents.get_training_results_documents()
    decision = Decision()

    df = decision.plagiarism_report_documents(lst)
    print(df.to_string())

    df.to_csv('documentos_sospechosos.csv', index=False)
    print('Resultados guardados en .csv como documentos_sospechosos.csv')

def print_auc():
    doc2vec = Processing(training_directory='./training_data',
                         test_directory='./test_data',
                         document_or_sentences='sentences',
                         lemmatize_or_stemming='lemmatize')

    doc2vec.train_model()
    training_results = doc2vec.get_training_results_sentences()

    validation_dictionary = {
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
        'org-002.txt': False,
        'org-005.txt': False,
        'org-012.txt': False,
        'org-020.txt': False,
        'org-026.txt': False,
        'org-040.txt': False,
        'org-047.txt': False,
        'org-056.txt': False,
        'org-059.txt': False,
        'org-061.txt': False,
        'org-065.txt': False,
        'org-071.txt': False,
        'org-081.txt': False,
        'org-084.txt': False,
        'org-090.txt': False,
        'org-093.txt': False,
        'org-100.txt': False,
        'org-001.txt': False,
        'org-110.txt': False
    }

    decision = Decision()

    confusion_matrix = decision.get_confusion_matrix(training_results,
                                                     validation_dictionary)

    auc = get_auc(confusion_matrix)
    print(confusion_matrix)
    print(auc)


def get_pdfs():
    decision = Decision()

    doc2vec = Processing(training_directory='./training_data',
                         test_directory='./test_data',
                         document_or_sentences='sentences',
                         lemmatize_or_stemming='lemmatize')

    doc2vec.train_model()

    doc2vec_documents = Processing(training_directory='./training_data',
                                   test_directory='./test_data',
                                   document_or_sentences='document',
                                   lemmatize_or_stemming='lemmatize')

    doc2vec_documents.train_model()

    for file in os.listdir('./test_data'):
        lst = doc2vec.get_most_similar_document_sentences(
            f'./test_data/{file}')

        top = doc2vec_documents.get_most_similar_documents(
            f'./test_data/{file}')

        result = decision.get_plagiarism_sentences(lst, top)

        generate_pdf(title=result['title'],
                     plagiarism_percent=result['plagiarism_percent'],
                     text=result['text'],
                     file_name=file)


if __name__ == '__main__':
    # plagiarism_types_report()

    # print_auc()

    get_pdfs()