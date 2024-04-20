# test_similarity

# Text Similarity Detection Project

This project is a Python-based application that uses Doc2Vec, a machine learning model, to detect the similarity between different text documents. The performance of the model is evaluated using the Area Under the Curve (AUC) metric.

## Project Overview

The project consists of three main components:

1. **Preprocessing**: This component is responible of cleaning all of our data and is used as a first step to prepare the data for the Doc2Vec model.

2. **Processing**: This component is responsible for training the Doc2Vec model on a given set of training data. It also provides functionality to retrieve the most similar sentences from a given document.

3. **Decision**: This component uses the output from the Processing component to make decisions about whether a text is plagiarized or not. It calculates a confusion matrix and the AUC to evaluate the performance of the model.

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
print(auc)
```

4. Example response
```
0.95
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

2. Initialize the Processing class with documents to get the most similar documents
```python
doc2vecdocuments = Processing(training_directory='../training_data',
                                  test_directory='../test_data',
                                  document_or_sentences='document',
                                  lemmatize_or_stemming='lemmatize')
doc2vecdocuments.train_model()
```

3. Get the top most similar documents
```python
top = doc2vecdocuments.get_most_similar_documents('../test_data/FID-01.txt')
```

4. Get the most similar document sentences which will return a list of the most similar sentences of an specific document

```python
lst = doc2vec.get_most_similar_document_sentences(
        '../test_data/FID-01.txt')
```

5. Print all the sentences along with if they were plagiarized or not.

```python
from decision import Decision

decision = Decision()

print(decision.get_plagiarism_sentences(lst, top))
```

6. Example response

```
PLAGIARISM DETECTED

Sentence: This article delves into the intricacies of adaptive fuzzy event-triggered formation tracking control for nonholonomic multirobot systems characterized by infinite actuator faults and range constraints.' || presents plagiarism from  'org-076.txt' sentence 'ï»¿This article delves into the intricacies of adaptive fuzzy event-triggered formation tracking control for nonholonomic multirobot systems characterized by infinite actuator faults and range constraints.'

 Plagiarized Sentence: Traditional cheating detection methods have many disadvantages, such as difficult to detect covert equipment cheating, multi-source cheating, difficult to distinguish plagiarists from plagiarists, difficult to distinguish plagiarists from victims, or plagiarism from coincidences. || does not present plagiarism

 Sentence: 'To address these issues, we leverage the power of fuzzy logic systems (FLSs) and employ adaptive methods to approximate unknown nonlinear functions and uncertain parameters present in robotic dynamics.' || presents plagiarism from  'org-076.txt' sentence 'To address these issues, we leverage the power of fuzzy logic systems (FLSs) and employ adaptive methods to approximate unknown nonlinear functions and uncertain parameters present in robotic dynamics.'

 Sentence: 'In the course of information exploration, the problems of collision avoidance and connectivity maintenance are ever present due to limitations of distance and visual fields.' || presents plagiarism from  'org-076.txt' sentence 'In the course of information exploration, the problems of collision avoidance and connectivity maintenance are ever present due to limitations of distance and visual fields.'

 Plagiarized Sentence: In this paper, the concept of knowledge point mastery Index is introduced to measure studentsâ mastery of a certain knowledge point, and a test method of cheating based on improved cognitive diagnostic model is proposed. || does not present plagiarism

 Sentence: 'Furthermore, to reduce the number of controller executions and compensate for any effect arising from infinite actuator failures, robots engage with their leader at the moment of actuator faults using fewer network communication resources yet maintain uninterrupted tracking of the desired trajectory generated by the leader.' || presents plagiarism from  'org-076.txt' sentence 'Furthermore, to reduce the number of controller executions and compensate for any effect arising from infinite actuator failures, robots engage with their leader at the moment of actuator faults using fewer network communication resources yet maintain uninterrupted tracking of the desired trajectory generated by the leader.'

 Sentence: 'We guarantee that all signals are semi-global uniformly ultimately bounded (SGUUB).' || presents plagiarism from  'org-076.txt' sentence 'We guarantee that all signals are semi-global uniformly ultimately bounded (SGUUB).'

 Plagiarized Sentence: Ultimately, we demonstrate the practical feasibility of the ETFT control strategy for nonholonomic multirobot systems.The experiments show that the precision and recall rate of this method are significantly higher than those of the method based on the false-same rate, the method based on the false-same rate and the right-same rate and the method based on the Person-Fit index. || does not present plagiarism

 Plagiarism percentage: 
0.5103433775786863

 Most similar document(s): 
org-076.txt with similarity: 0.85```