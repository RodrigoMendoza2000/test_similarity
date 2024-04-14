from nltk.tokenize import word_tokenize
import pandas as pd
import os
from preprocessing import Preprocessing
import nltk
nltk.download('punkt')
# https://carpentries-incubator.github.io/python-text-analysis/09-wordEmbed_train-word2vec/index.html


class Word2VecTraining:

    def __init__(self, directory):
        self.text_dataframe = None
        self.directory = directory
        self.text_files = os.listdir(f'../{directory}')
        self.preprocessing = Preprocessing()

    def files_to_dataframe(self):
        dic = {'file_name': [], 'file_text': []}
        for file_name in self.text_files:
            with open(f'../{self.directory}/{file_name}', encoding='ISO-8859-1') as file:
                dic['file_name'].append(file_name)
                print(file_name)
                file_text = " ".join(file.readlines())
                preprocessed_file_text = self.preprocessing.transform_prompt(file_text)
                dic['file_text'].append(preprocessed_file_text)

        self.text_dataframe = pd.DataFrame.from_dict(dic)


if __name__ == '__main__':
    #text_files_list = os.listdir('../training_data')
    #with open('../training_data/org-023.txt', 'r', encoding='ISO-8859-1') as f:
    #    print(f.readlines())
    w = Word2VecTraining('training_data')
    w.files_to_dataframe()


