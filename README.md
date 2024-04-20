# test_similarity

# Text Similarity Detection Project

This project is a Python-based application that uses Doc2Vec, a machine learning model, to detect the similarity between different text documents. The performance of the model is evaluated using the Area Under the Curve (AUC) metric.

## Project Overview

The project consists of two main components:

1. **Processing**: This component is responsible for training the Doc2Vec model on a given set of training data. It also provides functionality to retrieve the most similar sentences from a given document.

2. **Decision**: This component uses the output from the Processing component to make decisions about whether a text is plagiarized or not. It calculates a confusion matrix and the AUC to evaluate the performance of the model.

## Getting Started

These instructions will guide you on how to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project is developed using Python and requires several libraries. The required libraries are listed in the `requirements.txt` file. You can install them using pip:
```bash
pip install -r requirements.txt
````
### Usage
Follow these steps to obtain the AUC of the model:
1. Initialize the Processing class and train the model:

```python
from processing import Processing

doc2vec = Processing(training_directory='../training_data',
                     test_directory='../test_data',
                     document_or_sentences='sentences',
                     lemmatize_or_stemming='lemmatize')
doc2vec.train_model()
```

2. Get the training results
```python
training_results = doc2vec.get_training_results()
```

3. Initialize the Decision class and get the confusion matrix and AUC:

```python
from decision import Decision

decision = Decision()
confusion_matrix = decision.get_confusion_matrix(training_results, validation_dictionary)
auc = get_auc(confusion_matrix)
```

Follow these steps to Get the plagiarism sentences which will return a string of the most similar sentences in the training data. As well as the plagiarism percentage for the whole document.

1. Initialize the Processing class and train the model:

```python
from processing import Processing

doc2vec = Processing(training_directory='../training_data',
                     test_directory='../test_data',
                     document_or_sentences='sentences',
                     lemmatize_or_stemming='lemmatize')
doc2vec.train_model()
```

2. Get the most similar document sentences which will return a list of the most similar sentences of an specific document

```python
lst = doc2vec.get_most_similar_document_sentences(
        '../test_data/FID-01.txt')
```

3. Print all the sentences along with if they were plagiarized or not.

```python
from decision import Decision

decision = Decision()

print(decision.get_plagiarism_sentences(lst))
```