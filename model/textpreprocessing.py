# ---------------------------------------------------------------
# Preporcessing class
# Author: Diego Yunoe Sierra Díaz
# Last modified: 21/04/2024
# ---------------------------------------------------------------

from nltk.corpus import stopwords
from nltk import download
from nltk.stem import SnowballStemmer
import re
from nltk.stem import WordNetLemmatizer
import inflect
import num2words
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import os

download('punkt', quiet=True)
download('stopwords', quiet=True)
ENGLISH_STOPWORDS = stopwords.words("english")

# Regex compilations for additional performance
EMAIL_PATTERN = re.compile(r"\S*@\S*\s?")
MENTION_PATTERN = re.compile(r"@\S*\s?")
URL_PATTERN = re.compile(r"http\S+|www.\S+")
SYMBOL_PATTERN = re.compile(r"[^\w\sñ]")
SPECIAL_CASE_1 = re.compile(r"\nd")
SPECIAL_CASE_2 = re.compile(r"\n")
HASTAG_PATTERN = re.compile(r"\B(\#[a-zA-Z]+\b)(?!;)")
WORD_LENGTH_PATTERN = re.compile(r"\b(?!\d)\w{1,2}\b")
A_ACCENT_PATTERN = re.compile(r"[áàäâ]")
E_ACCENT_PATTERN = re.compile(r"[éèëê]")
I_ACCENT_PATTERN = re.compile(r"[íìïî]")
O_ACCENT_PATTERN = re.compile(r"[óòöô]")
U_ACCENT_PATTERN = re.compile(r"[úùüû]")
ALPHA_NUMERIC_PATTERN = re.compile(r"[^a-zA-Z\d|\s|ñ]")


class Preprocessing:
    """
        A class for text preprocessing tasks including removal of stopwords,
        stemming/lemmatization, formatting removal, and tokenization.

        Attributes:
            prompt (str): The text to be preprocessed.

        Methods:
            transform_prompt(prompt, tokenize=True, lemmatize_or_stemming=
            'lemmatize'):
                Performs all preprocessing steps on the given text.
                Arguments:
                    prompt (str): The text to be preprocessed.
                    tokenize (bool): Whether to tokenize the text or not.
                    lemmatize_or_stemming (str): The method to use for
                    lemmatization or stemming.
                        Possible values: 'lemmatize', 'stem'

            __tokenize():
                Tokenizes the text.

            __strip_formatting():
                Removes special characters and certain patterns from the text.

            __stem_prompt():
                Stems the text using SnowballStemmer.

            __lemmatize_prompt():
                Lemmatizes the text using WordNetLemmatizer.

            __remove_stopwords():
                Removes stopwords from the text.

            __convert_to_numeric():
                Converts all numbers to their word format.

            __replace_words():
                Replaces certain words with other words defined in a file.
                ex. AI -> Artificial intelligence
    """

    def __init__(self):
        self.prompt = ""

    def transform_prompt(self,
                         prompt: str,
                         lemmatize_or_stemming: str = 'lemmatize'
                         ) -> str:
        if lemmatize_or_stemming not in ['lemmatize', 'stem']:
            raise ValueError(
                "Invalid value for lemmatize_or_stemming. "
                "Possible values: 'lemmatize', 'stem'"
            )
        self.prompt = prompt.lower()
        # self.__convert_to_numeric()
        self.__replace_words()
        # sself.prompt)
        self.__remove_stopwords()
        self.__strip_formatting()
        if lemmatize_or_stemming == 'lemmatize':
            self.__lemmatize_prompt()
        elif lemmatize_or_stemming == 'stem':
            self.__stem_prompt()

        return self.prompt

    def tokenize(self, prompt: str) -> list[str]:
        return word_tokenize(prompt)

    def __strip_formatting(self) -> None:
        replace_to_blank = [
            EMAIL_PATTERN,
            MENTION_PATTERN,
            URL_PATTERN,
            SYMBOL_PATTERN,
            SPECIAL_CASE_1,
            SPECIAL_CASE_2,
            HASTAG_PATTERN,
            WORD_LENGTH_PATTERN
        ]

        for pattern in replace_to_blank:
            self.prompt = pattern.sub("", self.prompt)

    def __stem_prompt(self) -> None:
        stemmer = SnowballStemmer("english")
        self.prompt = stemmer.stem(self.prompt)

    def __lemmatize_prompt(self) -> None:
        wnl = WordNetLemmatizer()
        words = re.findall(r"\w+", self.prompt)

        self.prompt = " ".join(
            [
                wnl.lemmatize(word) for word in words
            ]
        )

    def __remove_stopwords(self) -> None:
        words = re.findall(r"\w+", self.prompt)
        important_words = (
            word for word in words if word not in ENGLISH_STOPWORDS
        )
        self.prompt = " ".join(important_words)

    # Convert all numbers to their word format ex. 42 -> forty-two
    def __convert_to_numeric(self) -> None:
        new_prompt = []
        words = re.findall(r"\w+", self.prompt)
        for word in words:
            try:
                new_prompt.append(num2words.num2words(word))
            except Exception as e:
                new_prompt.append(word)

        self.prompt = " ".join(new_prompt)

    def __replace_words(self) -> None:
        word_replaces = {}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'list_of_replaces.txt')
        with open(path, 'r') as file:
            for line in file:
                line_splitted = line.split()
                key = line_splitted[0]
                val = " ".join([w for w in line_splitted[1:]])
                word_replaces[key] = val
        for replace_from, replace_to in word_replaces.items():
            # print(replace_from, ' ', replace_to)
            self.prompt = re.sub(r"\b" + replace_from + r"\b", replace_to,
                                 self.prompt)



if __name__ == "__main__":
    text = r"Artificial intelligence (AI) mirrors human intelligence by " \
           r"enabling machines to execute tasks " \
           "traditionally handled by humans, leveraging technologies like " \
           "machine learning and neural networks. While " \
           "AI systems can adapt and improve through learning, concerns " \
           "persist regarding ethical implications such " \
           "as privacy infringement and biases in decision-making. 42 As AI " \
           "continues to evolve, responsible " \
           "development practices become imperative to navigate its " \
           " societal impact."

    preprocesser = Preprocessing()

    new_text = preprocesser.transform_prompt(text)

    print(new_text)

    with open(f'../training_data/org-007.txt', 'r',
              encoding='ISO-8859-1') as file:
        line_number = 1
        lines = " ".join(file.readlines())
        sentences = sent_tokenize(lines)
        for sentence in sentences:
            print(preprocesser.transform_prompt(sentence))
