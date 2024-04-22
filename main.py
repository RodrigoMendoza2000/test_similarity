from model.processing import Processing
from model.decision import Decision, get_auc

doc2vec = Processing(training_directory='./training_data',
                     test_directory='./test_data',
                     document_or_sentences='sentences',
                     lemmatize_or_stemming='lemmatize')

doc2vec.train_model()
training_results = doc2vec.get_training_results()

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

decision = Decision()

confusion_matrix = decision.get_confusion_matrix(training_results,
                                                 validation_dictionary)

auc = get_auc(confusion_matrix)
print(auc)