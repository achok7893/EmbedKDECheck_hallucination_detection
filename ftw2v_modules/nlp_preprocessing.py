import pandas as pd
import re
from nltk.tokenize import word_tokenize
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
import spacy
import numpy as np
import multiprocessing as mp
import nltk
import inflect
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import multiprocessing
from gensim.models.fasttext import FastText
import logging
from numpy.linalg import norm
import ftw2v_modules.bert_util as bert_utl
from functools import lru_cache



def cosine(A, B):
    cosine = np.dot(A, B)/(norm(A)*norm(B))
    return cosine


nlp = spacy.load('fr_core_news_md')
POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
TAG_LIST = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ",", "$", "#", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NIL", "NN", "NNP", "NNPS", "NNS",
            "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "ADD", "NFP", "GW", "XX", "BES", "HVS", "_SP"]
NER_LIST = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
            "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
# DEP_LIST = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]


def check_match(regex, input_string:str):
    trimmed_input_strip = input_string.strip()
    if re.match(regex, trimmed_input_strip):
        return True
    else:
        return False

def check_several_matches(list_regex_patterns, input_string:str):
    trimmed_input_strip = input_string.strip()
    l_checks = [check_match(regex_pattern, trimmed_input_strip) for regex_pattern in list_regex_patterns]
    return any(l_checks)



def lev_dist(a, b):
    '''
    This function will calculate the levenshtein distance between two input
    strings a and b
    
    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare
        
    returns:
        This function will return the distnace between string a and b.
        
    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    '''
    
    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)

def word_sim_lev_dist(word1, word2):
    lev_distance = lev_dist(word1, word2)
    return lev_distance/min(len(word1), len(word2))

def parallelize_dataframe(df, func):
    num_processes = mp.cpu_count()
    df_split = np.array_split(df, num_processes)
    with mp.Pool(num_processes) as p:
        df = pd.concat(p.map(func, df_split))
    return df


def parallelize_function(df):
    df["contenu_token"] = df["contenu"].apply(function_to_apply_for_parallel)
    return df


def function_to_apply_for_parallel(x):
    nlp = spacy.load('fr_core_news_md')
    return lemmatize_with_spacy_model(nlp, x)


def lemmatize_df_col_with_spacy_model(df):
    """Lemmatize every string row

    Args:
        df (pd.DataFrame): _description_
        nlp (spacy model): _description_

    Returns:
        (pd.DataFrame)
    """

    return parallelize_dataframe(df, parallelize_function)


def lemmatize_with_spacy_model(nlp, text: str) -> list:
    """Lemmatize string using spacy nlp model.

    Args:
        nlp (_type_): spacy nlp model.
        text (str): string.

    Returns:
        list: list of tokens
    """
    doc = nlp(text)
    res = []
    for token in doc:
        res.append(token.lemma_)
    return res


def remove_accents(x: str) -> str:
    """remove accents from string

    Args:
        x (str): _description_

    Returns:
        str: _description_
    """
    t = x
    dc_accents = {"e": ["é", "è", "ê"], "a": [
        "à"], "u": ["ù", "û"], "i": ["î"]}
    for case in list(dc_accents.items()):
        remplacant = case[0]
        for carac in case[1]:
            t = t.replace(carac, remplacant)
    return t


def clean_docs_old1(df0: pd.DataFrame,
                    col: str = "contenu",
                    dc_replace={" ": "_-:,;!?'/’+-#()[]|{}\"%*•<>"}) -> pd.DataFrame:
    """_summary_

    Args:
        df0 (pd.DataFrame, optional): DataFrame containing col to be cleaned
        col (str, optional): Defaults to "contenu". The column to clean
        dc_replace (dict, optional): Defaults to {" ": "_-:,;!?'/’+-#()[] | {}\"%*•<>"}. The chars to be replaces by the dict key

    Returns:
        pd.DataFrame: The cleaned dataframe
    """
    df = df0.copy()
    for case in list(dc_replace.items()):
        for car in case[1]:
            df.loc[:, col] = df.loc[:, col].str.replace(
                car, case[0], regex=False)

    df.loc[:, col] = df.loc[:, col].str.replace("[0-9]", " ", regex=True)
    df.loc[:, col] = df.loc[:, col].apply(
        lambda x: re.sub("\s\s+", " ", str(x)))
    df.loc[:, col] = df.loc[:, col].str.lower().str.strip()

    return df


def clean_docs_old2(df0: pd.DataFrame,
                    col: str = 'contenu',
                    str_replace={" ": ["<br>"]},
                    dc_replace={" ": "_;!?'’+-#()[]|{}\"%*•<>"},
                    remove_numbers=False,
                    lower_str=True,
                    add_stop_words=["d", "n", "qu"],
                    remove_stop_words=True):
    """_summary_

    Args:
        df0 (_type_, optional): _description_. Defaults to 'contenu', str_replace={" ": ["<br>"]}, dc_replace={" ": "_;!?'’+-#()[] | {}\"%*•<>"}.
        remove_numbers (bool, optional): _description_. Defaults to False.
        lower_str (bool, optional): _description_. Defaults to True.
        add_stop_words (list, optional): _description_. Defaults to ["d", "n", "qu"].
        remove_stop_words (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    df = df0.copy()

    for case in list(str_replace.items()):
        for car in case[1]:
            df.loc[:, col] = df.loc[:, col].str.replace(
                car, case[0], regex=False)

    for case in list(dc_replace.items()):
        for car in case[1]:
            df.loc[:, col] = df.loc[:, col].str.replace(
                car, case[0], regex=False)

    if remove_numbers:
        df.loc[:, col] = df.loc[:, col].str.replace("[0-9]", " ", regex=True)

    df.loc[:, col] = df.loc[:, col].apply(
        lambda x: re.sub("\s\s+", " ", str(x)))

    if lower_str:
        df.loc[:, col] = df.loc[:, col].str.lower().str.strip()

    if remove_stop_words:
        stop_words = list(fr_stop) + add_stop_words
        df.loc[:, col] = df.loc[:, col].apply(lambda x: " ".join(
            [word for word in word_tokenize(x) if not word in stop_words]))

    return df


def extract_sub_regexe(df0: pd.DataFrame, col: str = "contenu", substitute=False) -> pd.DataFrame:
    """Extract dates, measures, units and values.

    Args:
        df0 (pd.DataFrame): _description_
        col (str, optional): _description_. Defaults to "contenu".
        substitute (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    df = df0.copy()

    mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
            "aout", "août", "septembre", "novembre",
            "octobre", "décembre", "decembre", "fevrier"]

    un_mois = "|".join(mois)

    reg_dates = f"(?:(?<=en )\d\u007b4\u007d|(?<=le )[0-9]+/[0-9]+(?:/[0-9]+|[ ]*)|[0-9]*[ ]*(?:{un_mois})[ ]*\d\u007b4\u007d|[0-9]*[ ]+(?:{un_mois})[ .,:]+|[0-9]\u007b1,2\u007d/[0-9]\u007b1,2\u007d/[0-9]\u007b2,4\u007d)"

    df.loc[:, "[DATE]"] = df.loc[:, col].apply(lambda x: [(
        i.group(0), i.start(), i.end()) for i in re.finditer(reg_dates, x.lower())])

    if substitute:
        df.loc[:, col] = df.loc[:, col].apply(
            lambda x: re.sub(reg_dates, " [DATE]", x))

    regexs = {
        "[BIOAI]": "\([^\(\)]+\)",
        "[UNIT]": "(?:[^ :!]*(?:mol|MOL)[ ]*[\/][ ]*[a-zA-Z]+|[a-zA-Z]+[0-9]*[/][a-zA-Z]+[0-9]*|(?<= )[mk]*[gm](?=[ .,])|[a-zA-Z]+[ ]*/[ ]*(?:24|8)[ ]*[h]*|[/][a-zA-Z]+[0-9]*|°[cC])",
        "[BIOVAL]": "(?:(?<=: )[0-9]+[,]*[0-9]*|[0-9]+[,][0-9]+|(?<=[:.,\- ])[0-9]+[,]*[0-9]*[ ]*[%]|[0-9]+[ ]*/[ ]*[0-9]+|[0-9]+[ ]*(?=\[UNIT\]))"
    }

    for case in list(regexs.items()):
        df.loc[:, case[0]] = df.loc[:, col].apply(
            lambda x: [(i.group(0), i.start(), i.end()) for i in re.finditer(case[1], x)])
        if substitute:
            df.loc[:, col] = df.loc[:, col].apply(
                lambda x: re.sub(case[1], case[0], x))

    return df


def clean_docs(df0: pd.DataFrame, col: str = "contenu",
               str_replace: dict = {" ": ["<br>"]},
               dc_replace: dict = {" ": "_;!?'’+-#()[]|{}\"%*•<>"},
               remove_numbers: bool = False, lower_str: bool = True,
               add_stop_words: list = ["d", "n", "qu"],
               remove_stop_words: bool = False, add_spe_tokens: bool = False)\
        -> pd.DataFrame:
    r"""Perform several cleaning operations by replacing special characters.

    lowering strings ...

    Args:
        df0 (pd.DataFrame): Dataset containing the column <col> to be cleaned.
        col (str, optional): the column name. Defaults to "contenu",
        str_replace (dict, optional): replaces strings by a character.
        Defaults to {" ": ["<br>"]}
        dc_replace (dict, optional):.
        Defaults to dict = {" ": "_;!?'’+-#()[] | {}\"%*•<>"}.
        remove_numbers (bool, optional): _description_. Defaults to False.
        lower_str (bool, optional): _description_. Defaults to True.
        add_stop_words (list, optional): _description_.
        Defaults to ["d", "n", "qu"].
        remove_stop_words (bool, optional): _description_. Defaults to False.
        add_spe_tokens (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    # dc_replace={" ":"_;!?'’+-#()[]|{}\"%*•<>"}
    # str_replace={" ": ["<br>"]}
    df = df0.copy()

    for case in list(str_replace.items()):
        for car in case[1]:
            df.loc[:, col] = df.loc[:, col].str.replace(
                car, case[0], regex=False)

    for case in list(dc_replace.items()):
        for car in case[1]:
            df.loc[:, col] = df.loc[:, col].str.replace(
                car, case[0], regex=False)

    if remove_numbers:
        df.loc[:, col] = df.loc[:, col].str.replace("[0-9]", " ", regex=True)

    df.loc[:, col] = df.loc[:, col].apply(
        lambda x: re.sub("\s\s+", " ", str(x)))

    if lower_str:
        df.loc[:, col] = df.loc[:, col].str.lower().str.strip()

    if remove_stop_words:
        stop_words = list(fr_stop) + add_stop_words
        df.loc[:, col] = df.loc[:, col].apply(lambda x: " ".join(
            [word for word in word_tokenize(x) if not word in stop_words]))

    return df


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('french'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize_text_for_ml(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    # words = remove_stopwords(words)
    # words = stem_words(words)
    # words = lemmatize_verbs(words)
    return words


def lemmatize_with_spacy_model(nlp, text: str, return_list=True) -> list:
    """Lemmatize text with spacy model
    """
    doc = nlp(text)
    res = []
    for token in doc:
        res.append(token.lemma_)
    if return_list:
        return res
    else:
        return " ".join(res)


def tokenize(text):
    return nltk.word_tokenize(text)


def process_text_for_classification(x: str, nlp=nlp) -> str:
    x = " ".join(normalize_text_for_ml(
        lemmatize_with_spacy_model(nlp, x))).strip()
    x = re.sub("\\s+", " ", x)
    return x


def process_df_for_ml(df, source, nlp=nlp, target=None,  nb_folds=None):
    df.loc[:, "text"] = df.loc[:, source].apply(
        lambda x: process_text_for_classification(x, nlp))

    if not (nb_folds is None):
        df.loc[:, "folds"] = np.random.randint(0, nb_folds, df.shape[0])

    if not (target is None):
        df.loc[:, target] = df.loc[:, target].astype(str)
        le = LabelEncoder()
        df['label'] = le.fit_transform(df.loc[:, target])
        # df.loc[:, "label"] = df.loc[:,target].apply(lambda x: 1 if x==1 else 0 if x==0 else -1)
        # df.loc[:, "label"] = df.loc[:,target].apply(lambda x: 1 if x==1 else 0)
    return df


def get_embeddings(x_train, fast_text=True,
                   w2v_params: dict = {"min_count": 10,
                                       "window": 8,
                                       "vector_size": 300,
                                       "sample": 6e-5,
                                       "alpha": 0.03,
                                       "min_alpha": 0.0007,
                                       "negative": 20}) -> dict:
    """Get word embeddings.

    Args:
        x_train (list): list of strings.
        fast_text (bool, optional): Either use fast text or w2v . Defaults to True.
        w2v_params (_type_, optional): parameters to use. Defaults to {"min_count": 10, "window": 8, "vector_size": 300, "sample": 6e-5, "alpha": 0.03, "min_alpha": 0.0007, "negative": 20}.

    Returns:
        dict: containing the trained models
    """
    sent = [row.split(' ') for row in x_train]
    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
    w2v_params["workers"] = cores - 1

    if fast_text:
        w2v_model = FastText(**w2v_params)
    else:
        w2v_model = Word2Vec(**w2v_params)

    w2v_model.build_vocab(sent, progress_per=10000)
    w2v_model.train(sent, total_examples=w2v_model.corpus_count,
                    epochs=30, report_delay=1)

    return {"word_embeddings": w2v_model.wv.vectors,
            "vocabulary": w2v_model.wv.key_to_index,
            "w2v_model": w2v_model}


def train_glove_model(df0: pd.Series,
                      nb_dim=100,
                      learning_rate=0.05,
                      epochs=10,
                      window=10):
    """Train glove model on pandas series containing lines.
    To install glove on linux ubuntu. Enter the following commands:
    - sudo apt-get install python3-dev
    - pip install glove-python-binary
    """

    from glove import Corpus, Glove
    sentences = []
    for line in df0.to_list():
        # remove punctuation
        # line = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',line).strip()
        # tokenizer
        tokens = re.findall(r'\b\w+\b', line)
        if len(tokens) > 1:
            sentences.append(tokens)
    corpus = Corpus()
    corpus.fit(sentences, window=window)
    glove = Glove(no_components=nb_dim, learning_rate=learning_rate)
    glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')

    return {"w2v_model": glove,
            "vocabulary": glove.dictionary,
            "word_embeddings": glove.word_vectors}


def process_and_get_word_embeddings(df0, source, nlp=nlp, target=None,
                                    nb_folds=None, fold_to_exclude=None,
                                    get_embedding=True,
                                    fast_text=True,
                                    w2v_params: dict = {"min_count": 10,
                                                        "window": 8,
                                                        "vector_size": 300,
                                                        "sample": 6e-5,
                                                        "alpha": 0.03,
                                                        "min_alpha": 0.0007,
                                                        "negative": 20}):
    """Process and get word embeddings.

    Args:
        df0 (pd.DataFrame): dataset containing the corpus
        source (str): column name containing the corpus
        nlp (spacy model): defaults to news-fr-core
        target (str, optional): column containing the label. Defaults to None.
        nb_folds (_type_, optional): _description_. Defaults to None.
        fold_to_exclude (_type_, optional): _description_. Defaults to None.
        get_embedding (bool, optional): _description_. Defaults to True.
        fast_text (bool, optional): _description_. Defaults to True.
        w2v_params (_type_, optional): _description_. Defaults to {"min_count": 10, "window": 8, "vector_size": 300, "sample": 6e-5, "alpha": 0.03, "min_alpha": 0.0007, "negative": 20}.

    Returns:
        _type_: _description_
    """
    df = df0.copy()
    df = process_df_for_ml(df, source, nlp, target=target,  nb_folds=nb_folds)
    if get_embedding:
        if not (nb_folds is None or fold_to_exclude is None):
            x_train = df.loc[df.folds != fold_to_exclude, "text"]
        else:
            x_train = df.loc[:, "text"]

        w2v = get_embeddings(x_train, fast_text=fast_text,
                             w2v_params=w2v_params)

        return {"df": df,
                "w2v": w2v}
    else:
        return {"df": df}


def w2v_get_word_vector_and_similarities(w2v, word1, topn=10, word2=None):
    """USE W2V model to get word embedding and similarity results.

    Args:
        w2v (_type_): _description_
        topn (int, optional): _description_. Defaults to 10.
        word1 (string): main word. Defaults to None.
        word2 (string, optional): main with which to compare. Defaults to None.
    """
    res = {}
    if word1:
        res["vector"] = w2v.wv[word1]
        res["most_similar"] = w2v.wv.most_similar(word1, topn=topn)
        res["most_opposite"] = w2v.wv.most_similar(negative=[word1], topn=topn)

    if word1 and word2:
        res["similarity"] = w2v.wv.similarity(word1, word2)

    return res


def get_sentence_embedding_and_pos_tag(text: str, w2v: FastText, nlp: spacy.lang.fr.French = nlp) -> pd.DataFrame:
    """Create a vector per token for training.

    Args:
        text (str): text to vectorize
        w2v (FastText): Word 2 vec trained model
        nlp (spacy.lang.fr.French): Spacy model for tokenization, pos and dep tagging


    Returns:
        pd.DataFrame: dataframe containing the vecotrization per token
    """
    doc = nlp(text)
    DEP_LIST = [label for label in nlp.get_pipe("parser").labels]
    x_tokens = []
    cols_emb = ["dim_" + str(i) for i in range(w2v.vector_size)]
    for token in doc:
        s_append = pd.Series(list(
            w2v.wv[token.text]) + [token.pos_, token.dep_], index=cols_emb + ["pos", "dep"]).to_frame()
        x_tokens.append(s_append.copy())
    x = pd.concat(x_tokens, axis=1, ignore_index=True).T
    x.pos = pd.Categorical(x.pos, categories=POS_LIST)
    x.dep = pd.Categorical(x.dep, categories=DEP_LIST)
    x = pd.get_dummies(x, columns=["dep"], prefix='dep')
    x = pd.get_dummies(x, columns=["pos"], prefix='pos')
    return x


def get_word_embedding(word: str, res_w2v, res_fasttext, fast_text_embeddings):
    """Give the embedding based on two different embedding approaches.
    Params:
        word (str): the word for which the embedding is needed

    """
    vec_ = None
    try:
        vec_ = res_w2v['w2v_model'].wv[word]
    except:
        similar_words = [i[0]
                         for i in res_fasttext['w2v_model'].wv.most_similar(word)]
        for word_i in similar_words:
            try:
                vec_ = res_w2v['w2v_model'].wv[word_i]
                break
            except:
                continue
        if vec_ is None:

            fast_text_word_array = res_fasttext['w2v_model'].wv[word]
            target_word = fast_text_embeddings.apply(lambda x: cosine(
                x, fast_text_word_array), axis=1).sort_values(ascending=False).index[0]
            vec_ = res_w2v['w2v_model'].wv[target_word]

    return vec_


def get_word_embedding_w2v(word: str, res_w2v):
    """Give the embedding based on two different embedding approaches.
    Params:
        word (str): the word for which the embedding is needed

    """
    vec_ = None
    try:
        vec_ = res_w2v['w2v_model'].wv[word]
    except:
        vec_ = np.zeros(300).astype(float)

    return vec_

def get_word_embedding_glove_only(word: str, res_w2v):
    """Give the embedding based on two different embedding approaches.
    Params:
        word (str): the word for which the embedding is needed

    """
    vec_ = None
    try:
        vec_ = res_w2v['word_embeddings'][res_w2v['vocabulary'][word]]
    except:
        vec_ = np.zeros(300).astype(float)

    return vec_

def get_word_embedding_ft(word: str, res_fasttext):
    """Give the embedding of FastText model.
    Params:
        word (str): the word for which the embedding is needed

    """
    vec_ = res_fasttext['w2v_model'].wv[word]
    return vec_


def get_word_embedding_glove(word: str, res_w2v, res_fasttext, fast_text_embeddings):
    """Give the embedding based on two different embedding approaches.
    Params:
        word (str): the word for which the embedding is needed

    """
    vec_ = None
    try:
        vec_ = res_w2v['word_embeddings'][res_w2v['vocabulary'][word]]
    except:
        similar_words = [i[0]
                         for i in res_fasttext['w2v_model'].wv.most_similar(word)]
        for word_i in similar_words:
            try:
                vec_ = res_w2v['word_embeddings'][res_w2v['vocabulary'][word_i]]
                break
            except:
                continue
        if vec_ is None:

            fast_text_word_array = res_fasttext['w2v_model'].wv[word]
            target_word = fast_text_embeddings.apply(lambda x: cosine(
                x, fast_text_word_array), axis=1).sort_values(ascending=False).index[0]
            vec_ = res_w2v['word_embeddings'][res_w2v['vocabulary'][target_word]]

    return vec_


def get_sentence_embedding(sentence: str, res_w2v, res_fasttext, fast_text_embeddings):
    words = sentence.split()
    l_vecs = []
    for word in words:
        try:
            l_vecs.append(get_word_embedding(
                word, res_w2v, res_fasttext, fast_text_embeddings))
        except:
            continue
    return np.mean(l_vecs, axis=0)

def get_sentence_embedding_w2v(sentence: str, res_w2v):
    words = sentence.split()
    l_vecs = []
    for word in words:
        try:
            l_vecs.append(get_word_embedding_w2v(
                word, res_w2v))
        except:
            continue
    return np.mean(l_vecs, axis=0)

def get_sentence_embedding_ft(sentence: str, res_fasttext):
    words = sentence.split()
    l_vecs = []
    for word in words:
        try:
            l_vecs.append(get_word_embedding_ft(
                word, res_fasttext))
        except:
            continue
    return np.mean(l_vecs, axis=0)


def get_sentence_embedding_glove_only(sentence: str, res_w2v):

    words = sentence.split()
    l_vecs = []
    for word in words:
        try:
            l_vecs.append(get_word_embedding_glove_only(
                word, res_w2v))
        except:
            continue
    return np.mean(l_vecs, axis=0)




def get_sentence_embedding_glove(sentence: str, res_w2v, res_fasttext, fast_text_embeddings):

    words = sentence.split()
    l_vecs = []
    for word in words:
        try:
            l_vecs.append(get_word_embedding_glove(
                word, res_w2v, res_fasttext, fast_text_embeddings))
        except:
            continue
    return np.mean(l_vecs, axis=0)


def get_sentence_spe_embedding_and_pos_tag(text: str,
                                           res_w2v, res_fasttext,
                                           fast_text_embeddings,
                                           nlp: spacy.lang.fr.French = nlp) -> pd.DataFrame:
    """Create a vector per token for training.

    Args:
        text (str): text to vectorize
        w2v (FastText): Word 2 vec trained model
        nlp (spacy.lang.fr.French): Spacy model for tokenization, pos and dep tagging


    Returns:
        pd.DataFrame: dataframe containing the vecotrization per token
    """
    doc = nlp(text)
    DEP_LIST = [label for label in nlp.get_pipe("parser").labels]
    x_tokens = []
    cols_emb = ["dim_" + str(i) for i in range(300)]
    for token in doc:
        s_append = pd.Series(list(
            get_word_embedding(token.text, res_w2v,
                               res_fasttext, fast_text_embeddings)) +
                             [token.pos_, token.dep_], index=cols_emb +
                             ["pos", "dep"]).to_frame()
        x_tokens.append(s_append.copy())
    x = pd.concat(x_tokens, axis=1, ignore_index=True).T
    x.pos = pd.Categorical(x.pos, categories=POS_LIST)
    x.dep = pd.Categorical(x.dep, categories=DEP_LIST)
    x = pd.get_dummies(x, columns=["dep"], prefix='dep')
    x = pd.get_dummies(x, columns=["pos"], prefix='pos')
    return x


def get_sentence_glove_embedding_and_pos_tag(text: str,
                                             res_w2v, res_fasttext,
                                             fast_text_embeddings,
                                             nlp: spacy.lang.fr.French = nlp) -> pd.DataFrame:
    """Create a vector per token for training.

    Args:
        text (str): text to vectorize
        w2v (FastText): Word 2 vec trained model
        nlp (spacy.lang.fr.French): Spacy model for tokenization, pos and dep tagging


    Returns:
        pd.DataFrame: dataframe containing the vecotrization per token
    """
    doc = nlp(text)
    DEP_LIST = [label for label in nlp.get_pipe("parser").labels]
    x_tokens = []
    cols_emb = ["dim_" + str(i) for i in range(300)]
    for token in doc:
        s_append = pd.Series(list(
            get_sentence_embedding_glove(
                token.text, res_w2v, res_fasttext, fast_text_embeddings)
        ) +
            [token.pos_, token.dep_], index=cols_emb +
            ["pos", "dep"]).to_frame()
        x_tokens.append(s_append.copy())
    x = pd.concat(x_tokens, axis=1, ignore_index=True).T
    x.pos = pd.Categorical(x.pos, categories=POS_LIST)
    x.dep = pd.Categorical(x.dep, categories=DEP_LIST)
    x = pd.get_dummies(x, columns=["dep"], prefix='dep')
    x = pd.get_dummies(x, columns=["pos"], prefix='pos')
    return x


def get_sentence_bert_embedding_and_pos_tag(text: str,
                                            model, tokenizer,
                                            nlp: spacy.lang.fr.French = nlp) -> pd.DataFrame:
    """Create a vector per token for training.

    Args:
        text (str): text to vectorize
        w2v (FastText): Word 2 vec trained model
        nlp (spacy.lang.fr.French): Spacy model for tokenization, pos and dep tagging


    Returns:
        pd.DataFrame: dataframe containing the vecotrization per token
    """
    doc = nlp(text)
    DEP_LIST = [label for label in nlp.get_pipe("parser").labels]
    x_tokens = []
    cols_emb = ["dim_" + str(i) for i in range(768)]
    for token in doc:
        s_append = pd.Series(list(
            bert_utl.get_embedding_with_camembert(token.text,
                                                  model=model, tokenizer=tokenizer)) +
                             [token.pos_, token.dep_], index=cols_emb +
                             ["pos", "dep"]).to_frame()
        x_tokens.append(s_append.copy())
    x = pd.concat(x_tokens, axis=1, ignore_index=True).T
    x.pos = pd.Categorical(x.pos, categories=POS_LIST)
    x.dep = pd.Categorical(x.dep, categories=DEP_LIST)
    x = pd.get_dummies(x, columns=["dep"], prefix='dep')
    x = pd.get_dummies(x, columns=["pos"], prefix='pos')
    return x


def get_sentence_charbert_embedding_and_pos_tag(text: str,
                                                model, tokenizer,
                                                nlp: spacy.lang.fr.French = nlp) -> pd.DataFrame:
    """Create a vector per token for training.

    Args:
        text (str): text to vectorize
        w2v (FastText): Word 2 vec trained model
        nlp (spacy.lang.fr.French): Spacy model for tokenization, pos and dep tagging


    Returns:
        pd.DataFrame: dataframe containing the vecotrization per token
    """
    doc = nlp(text)
    DEP_LIST = [label for label in nlp.get_pipe("parser").labels]
    x_tokens = []
    cols_emb = ["dim_" + str(i) for i in range(768)]
    for token in doc:
        s_append = pd.Series(list(
            bert_utl.get_embedding_with_charbert(token.text,
                                                 model=model, tokenizer=tokenizer)) +
                             [token.pos_, token.dep_], index=cols_emb +
                             ["pos", "dep"]).to_frame()
        x_tokens.append(s_append.copy())
    x = pd.concat(x_tokens, axis=1, ignore_index=True).T
    x.pos = pd.Categorical(x.pos, categories=POS_LIST)
    x.dep = pd.Categorical(x.dep, categories=DEP_LIST)
    x = pd.get_dummies(x, columns=["dep"], prefix='dep')
    x = pd.get_dummies(x, columns=["pos"], prefix='pos')
    return x
