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
    :param original_text: The original text which is not plagiarzed
    :param plagiarized_text: The suspicious text which is most likely
    plagiarized
    :return: bool
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
    :param original_text:
    :param plagiarized_text:
    :return:
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
                            plagiarized_text: str,
                            cosine_similarity: int) -> bool:
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
                         cosine_similarity: int) -> list:
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

    if identify_insert_replace(original_text, plagiarized_text,
                               cosine_similarity):
        cambios.append("Insertar o reemplazar frases")

    # Parafraseo
    if not cambios:
        cambios.append("Parafraseo")

    # print(cambios)

    return cambios


if __name__ == '__main__':
    # Ejemplo de uso
    # original = "The main idea of this paper is the substantiation of the methodological approach to the assessment of personnel risks of enterprises based on the application of the fuzzy logic apparatus in order to identify the problems of personnel risk management and provide appropriate recommendations for their solution. The methodological basis of the study is the classic provisions and fundamental works of foreign and domestic scientists, statistical data, the results of our research into the problems of assessing personnel risks of enterprises. The methods of fuzzy set theory, comparative analysis, scientific abstraction, generalization of scientific experience of modern theoretical research, systemcomplex approach were used. The study proposed a methodological approach to assessing the level of personnel risks of an enterprise; numerical experiments were conducted on the basis of a group of construction equipment manufacturers. Analysis of the results of assessing the level of personnel risks of enterprises made it possible to identify the problems of managing personnel risks at enterprises Statement of a mathematical problem: the work considers hierarchical fuzzy data, namely: four groups of indicators for assessing the level of personnel risks (quantitative composition – F1, state of qualifications and intellectual potential – F2, staff turnover – F3, motivational system – F4), each of the indicators has a different number of fuzzy coefficients (there are twelve of them in the current work – vi , i=1÷12). Indicators are functions of fuzzy coefficients: F1 = r(v1, v2, v3); F2 = g(v4,v5, v6, v7); F3 = h(v8, v9, v10,); F4=q(v11, v12). As an output variable, there is a functional – an integrated indicator Int = f(F1, F2, F3, F4) of the personnel risk level, which, in turn, is also a fuzzy value. Here, the functions r, g, h, q, f are unknown functions of the given variables. We have expert evaluations of the change in all input data; as a rule, they vary within three terms: Low (I), Medium (G), High (E). Formalized information on each variable can be written as , then for a group of indicators we have: . Using a fuzzy system and performing calculations with its help requires the system to have the following structural elements: membership functions of input and output variables, a rule base, and an output mechanism. These structural elements are the components that will be built when designing a fuzzy system. The built mathematical model and the method of its formalization on the basis of FST make it possible to estimate the level of personnel risk at the enterprise, which enables further substantiation of a set of measures to increase the efficiency of its use. The constructed system of fuzzy logical inference can be considered intelligent as it uses elements of computational intelligence, in particular, the theory of fuzzy sets. The proposed methodological approach to assessing the level of personnel risks of enterprises based on the apparatus of fuzzy logic allows, in contrast to existing ones, to integrate the consideration of both qualitative and quantitative indicators when assessing the level of personnel risks and personnel movement indicators and to significantly increase the efficiency of decision-making under conditions of uncertainty and reduce costs in the event of adverse situations."
    # plagiarized = "The main idea of this paper is the substantiation of the methodological approach to the assessment of personnel risks of enterprises based on the application of the fuzzy logic apparatus in order to identify the problems of personnel risk management and provide appropriate recommendations for their solution. The methodological basis of the study is the classic provisions and fundamental works of foreign and domestic scientists, statistical data, the results of our research into the problems of assessing personnel risks of enterprises. The methods of fuzzy set theory, comparative analysis, scientific abstraction, generalization of scientific experience of modern theoretical research, systemcomplex approach were used. The study proposed a methodological approach to assessing the level of personnel risks of an enterprise; numerical experiments were conducted on the basis of a group of construction equipment manufacturers. Analysis of the results of assessing the level of personnel risks of enterprises made it possible to identify the problems of managing personnel risks at enterprises Statement of a mathematical problem: the work considers hierarchical fuzzy data, namely: four groups of indicators for assessing the level of personnel risks (quantitative composition – F1, state of qualifications and intellectual potential – F2, staff turnover – F3, motivational system – F4), each of the indicators has a different number of fuzzy coefficients (there are twelve of them in the current work – vi , i=1÷12). Indicators are functions of fuzzy coefficients: F1 = r(v1, v2, v3); F2 = g(v4,v5, v6, v7); F3 = h(v8, v9, v10,); F4=q(v11, v12). As an output variable, there is a functional – an integrated indicator Int = f(F1, F2, F3, F4) of the personnel risk level, which, in turn, is also a fuzzy value. Here, the functions r, g, h, q, f are unknown functions of the given variables. We have expert evaluations of the change in all input data; as a rule, they vary within three terms: Low (I), Medium (G), High (E). Formalized information on each variable can be written as , then for a group of indicators we have: . Using a fuzzy system and performing calculations with its help requires the system to have the following structural elements: membership functions of input and output variables, a rule base, and an output mechanism. These structural elements are the components that will be built when designing a fuzzy system. The built mathematical model and the method of its formalization on the basis of FST make it possible to estimate the level of personnel risk at the enterprise, which enables further substantiation of a set of measures to increase the efficiency of its use. The constructed system of fuzzy logical inference can be considered intelligent as it uses elements of computational intelligence, in particular, the theory of fuzzy sets. The proposed methodological approach to assessing the level of personnel risks of enterprises based on the apparatus of fuzzy logic allows, in contrast to existing ones, to integrate the consideration of both qualitative and quantitative indicators when assessing the level of personnel risks and personnel movement indicators and to significantly increase the efficiency of decision-making under conditions of uncertainty and reduce costs in the event of adverse situations.This paper substantiates a methodological approach to assessing personnel risks in enterprises using fuzzy logic. The goal is to identify and provide recommendations for managing personnel risk problems. The study's foundation includes classic provisions of foreign and domestic scientists, statistical data, and the team's own research. Methods such as fuzzy set theory, comparative analysis, scientific abstraction, generalization of modern theoretical research, and a system-complex approach were employed. The study proposes a methodological approach to assessing personnel risk levels, exemplified by a group of construction equipment manufacturers. It considers hierarchical fuzzy data comprising four groups of indicators: quantitative composition (F1), state of qualifications and intellectual potential (F2), staff turnover (F3), and motivational system (F4). Each indicator has fuzzy coefficients (vi, i=1÷12), and the output variable is an integrated indicator Int, representing the personnel risk level. The paper introduces unknown functions r, g, h, q, f as functions of fuzzy coefficients, reflecting expert evaluations of input data changes (Low, Medium, High). The study utilizes a fuzzy system with structural elements including membership functions, a rule base, and an output mechanism. This system allows for estimating personnel risk levels and justifying measures to enhance efficiency. The fuzzy logical inference system is considered intelligent as it employs computational intelligence elements, particularly fuzzy set theory. This methodological approach integrates qualitative and quantitative indicators, enhancing decision-making efficiency under uncertainty and reducing costs during adverse situations, distinguishing it from existing methods."
    # original = "The cat was chased by the dog."
    # plagiarized = "The cat is chasing the dog."
    # original = "In this context, the use of artificial empathy strategies is particularly interesting because of their potential to improve customer experiences affectively and socially."
    # plagiarized = "In this context, the use of artificial empathy strategies is of particular interest due to its potential in improving customer experiences affectively and socially."

    # tipo_cambio = identify_text_change(original, plagiarized)
    # print(f"Tipo de cambio de texto: {tipo_cambio}")
    plagiarized = "As of 2021, more than 30 countries have released national artificial intelligence (AI) policy strategies."
    original = 'As of 2021, national artificial intelligence (AI) policy strategies have been released by more than 30 countries.'

    print(identify_voice_change(original, plagiarized))
