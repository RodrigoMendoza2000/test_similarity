import difflib
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger', quiet=True)


def change_tense(sentence: str) -> dict:
    """
    This function receives a sentence and returns a dictionary with the
    present verbs and their frequency.
    :param sentence: The sentence to get the verbs from
    :return: dictionary{verb: frequency}
    """
    wordn = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    present_sentence = []
    for token, tag in tagged_tokens:
        if tag.startswith('V'):
            if tag is None:
                present_sentence.append(token)
            else:
                lemma = wordn.lemmatize(token, 'v')
                if lemma is None:
                    present_sentence.append(token)
                else:
                    present_sentence.append(lemma)
    return {x: present_sentence.count(x) for x in present_sentence}


def identify_time_change(original_text: str, plagiarized_text: str) -> bool:
    """
    This function receives two texts and returns whether there is a change in
    tense or not. Converts all the verbs in the text to their base form and
    compares them. If there is a 95% similarity in the verbs, then there is a
    change in tense.
    :param original_text: The original text which is not plagiarzed and is
    from our training data
    :param plagiarized_text: The plagiarized text which the AI model suspects
    and is from our test data
    :return: bool, true if there is a change in tense, false otherwise
    """
    original_verb_dict = change_tense(original_text)
    plagiarized_verb_dict = change_tense(plagiarized_text)

    # Calculate the total number of verbs in both dictionaries
    total_original_verbs = sum(original_verb_dict.values())
    total_plagiarized_verbs = sum(plagiarized_verb_dict.values())

    # Calculate the intersection of verbs between the two dictionaries
    intersection_verbs = set(original_verb_dict.keys()) & set(
        plagiarized_verb_dict.keys())

    # Calculate the total number of matching verbs
    total_matching_verbs = sum(min(original_verb_dict.get(verb, 0),
                                   plagiarized_verb_dict.get(verb, 0)) for verb
                               in intersection_verbs)

    # Calculate the similarity ratio
    similarity_ratio = total_matching_verbs / max(total_original_verbs,
                                                  total_plagiarized_verbs)

    # Check if the similarity ratio exceeds the threshold
    if similarity_ratio >= 0.9 \
            and not \
            identify_unordered_sentences(original_text, plagiarized_text) \
            and original_text != plagiarized_text:
        return True
    else:
        return False


def identify_voice_change(original_text: str, plagiarized_text: str) -> bool:
    # Tokenizar los textos
    tokens1 = nltk.word_tokenize(original_text)
    tokens2 = nltk.word_tokenize(plagiarized_text)

    # Etiquetar las partes del discurso
    pos_tags1 = nltk.pos_tag(tokens1)
    pos_tags2 = nltk.pos_tag(tokens2)

    # Identificar la voz en cada texto
    voz1 = "activa" if not identify_passive_voice(pos_tags1) else "pasiva"
    voz2 = "activa" if not identify_passive_voice(pos_tags2) else "pasiva"

    # Si las voces no son las mismas, entonces hay un cambio de voz
    if voz1 != voz2:
        return True
    else:
        return False


def identify_passive_voice(tags):
    # Buscar la estructura de la voz pasiva (verbo "to be" seguido de un
    # participio pasado)
    return any(
        t1[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and t2[1] == 'VBN'
        for t1, t2 in zip(tags, tags[1:]))


def identify_unordered_sentences(original_text: str,
                                 plagiarized_text: str) -> bool:
    """
    A text is considered to be unordered if the sentences are the same
    but in a different order.
    :param original_text: The original text which is not plagiarzed and is
    from our training data
    :param plagiarized_text: The plagiarized text which the AI model suspects
    and is from our test data
    :return: bool True if the text is unordered, False otherwise
    """
    # Tokenizar los textos
    tokens1 = sent_tokenize(original_text)
    tokens2 = sent_tokenize(plagiarized_text)

    for sentence in tokens1:
        if sentence not in tokens2:
            return False

    return True

    """# Remove repeated sentences
    tokens1 = list(set(tokens1))
    tokens2 = list(set(tokens2))

    joined_tokens_1 = " ".join(tokens1)
    joined_tokens_2 = " ".join(tokens2)

    if sorted(joined_tokens_1.split()) == sorted(joined_tokens_2.split()):
        return True
    else:
        return False"""


def identify_insert_replace(original_text: str,
                            plagiarized_text: str) -> bool:
    """
    A sentence is considered to have sentences inserted if the plagiarized
    text is the same as the original but sentences were added after or before
    the text was plagiarized. For this we compare whether the plagiarized text
    contains the original text and additionally the get_opcodes returns us that
    sentences have to be inserted to reach the text.

    A sentence is considered replaced if the plagiarized text contains a large
    number of sentences from the original text but some were replaced by
    others. For this we compare the sentences between both texts and if the
    plagiarized one contains the majority of the sentences then it is
    considered replaced.

    :param original_text: The original text which is not plagiarzed and is
    from our training data
    :param plagiarized_text: The plagiarized text which the AI model suspects
    and is from our test data
    :return: bool True if it identified the text
    was of type insert or replace, False otherwise
    """
    s = difflib.SequenceMatcher(None, original_text, plagiarized_text)
    plagiarized_text_sent = sent_tokenize(plagiarized_text)
    total_words = len(word_tokenize(plagiarized_text))

    percentage = 0
    for sentence in plagiarized_text_sent:
        if sentence in original_text:
            percentage += len(word_tokenize(sentence)) / total_words

    #if cosine_similarity > 0.8:
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'insert' and original_text in plagiarized_text:
            return True
        if tag in ['replace'] and percentage > 0.6:
            return True
    return False


def identify_text_change(original_text: str,
                         plagiarized_text: str,
                         cosine_similarity: float) -> list:
    """
    Obtain the type of changes that were identified in the text

    :param original_text: The original text which is not plagiarzed and is
    from our training data
    :param plagiarized_text: The plagiarized text which the AI model suspects
    and is from our test data
    :param cosine_similarity: The cosine similarity between the two texts
    :return: A list of the type of changes that were identified
    """
    cambios = []

    if identify_unordered_sentences(original_text,
                                    plagiarized_text):
        cambios.append("Desordenar las frases")
    # Insertar o reemplazar frases
    # if not cambios:

    if identify_voice_change(original_text,
                             plagiarized_text):
        cambios.append("Cambio de voz")

    if identify_time_change(original_text,
                            plagiarized_text):
        cambios.append("Cambio de tiempo")

    if identify_insert_replace(original_text, plagiarized_text):
        cambios.append("Insertar o reemplazar frases")

    # Parafraseo
    if not cambios:
        cambios.append("Parafraseo")

    # print(cambios)

    return cambios


if __name__ == '__main__':

    # tipo_cambio = identify_text_change(original, plagiarized)
    # print(f"Tipo de cambio de texto: {tipo_cambio}")
    plagiarized = "As of 2021, more than 30 countries have released " \
                  "national artificial intelligence (AI) policy strategies."
    original = 'As of 2021, national artificial intelligence (AI) policy ' \
               'strategies have been released by more than 30 countries.'

    print(identify_voice_change(original, plagiarized))
