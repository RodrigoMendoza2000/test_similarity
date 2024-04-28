# ---------------------------------------------------------------
# Main test file
# Author: Rodrigo Alfredo Mendoza España
# Last modified: 21/04/2024
# ---------------------------------------------------------------
import os
from model.processing import Processing
from model.decision import Decision, get_auc

doc2vec = Processing(training_directory='./training_data',
                     test_directory='./test_data',
                     document_or_sentences='sentences',
                     lemmatize_or_stemming='lemmatize')

doc2vec.train_model()
training_results = doc2vec.get_training_results_sentences()

doc2vec_documents = Processing(training_directory='./training_data',
                               test_directory='./test_data',
                               document_or_sentences='document',
                               lemmatize_or_stemming='lemmatize')
doc2vec_documents.train_model()
"""
for file_name in os.listdir(f'../test_data'):
    if file_name.endswith('.txt'):
        print(doc2vec_documents.get_most_similar_documents('../test_data/'+file_name))
"""

# 76 - FID-01.txt, 104 - FID-02.txt, 16 - FID-03.txt, 45 - FID-04.txt,
# 85 - FID-05.txt, 43 - FID-06.txt, 41 - FID-07.txt, 79 - FID-08.txt,
# 109 - FID-09.txt, 79 - FID-10.txt

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
"""import pandas as pd

documento_sospechoso = []
copia = []

documento_plagiado = []
porcentaje_plagio = []
print('Documento sospechoso\t| Copia\t| Documento Plagiado\t| % plagio')
for file_name in os.listdir(f'test_data'):
    result = ''
    with open(f'test_data/{file_name}', 'r',
              encoding='ISO-8859-1') as file:

        if ".txt" in file_name:
            top = doc2vec_documents.get_most_similar_documents(
                f'test_data/{file_name}')
            documento_sospechoso.append(file_name)
            result += f'{file_name}\t'

            if top != {}:
                first_key = next(iter(top))
                percentage = top[first_key]
                copia.append('Si')
                documento_plagiado.append(first_key)
                porcentaje_plagio.append(round(percentage,2))
                result += f'  \t\t\t  Si\t  {first_key}\t\t\t {round(percentage,2)}'

            else:
                result += '  \t\t\t  No\t  Ninguno\t\t'
                copia.append('No')
                documento_plagiado.append('Ninguno')
                porcentaje_plagio.append(None)
            print(result)

dict = {'Documento sospechoso': documento_sospechoso,
        'Copia': copia,
        'Documento Plagiado': documento_plagiado,
        '% plagio': porcentaje_plagio}"""

lst = doc2vec_documents.get_training_results_documents()

decision = Decision()

df = decision.plagiarism_report_documents(lst)
print(df.to_string())
# df = pd.DataFrame(dict)
# df.to_csv('documentos_sospechosos.csv', index=False)
# print('Resultados guardados en .csv como documentos_sospechosos.csv')

# decision = Decision()

# confusion_matrix = decision.get_confusion_matrix(training_results,
#                                                  validation_dictionary)

# auc = get_auc(confusion_matrix)
# print(confusion_matrix)
# print(auc)

# print(doc2vec_documents.get_most_similar_documents('test_data/FID-07.txt'))
