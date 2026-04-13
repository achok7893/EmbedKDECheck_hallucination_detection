import pandas as pd
import zipfile
import re
import pickle
import numpy as np

dc_correct = {"‰": "â", "Ž": "é", "™": "ô", "ˆ": "à", "ž": "û", "õ": "\'", "Ê": "s", "ƒ": "é", "": "é", "\x8e": "é", "\x88": "à", "\x89": "â", "\x99": "ô",
              "\x90": "ê", "\x8f": "è", "\x9e": "û"}


def standard_cleaning(df0: pd.DataFrame, input_col: str, output_col: str) -> pd.DataFrame:
    """Standard cleaning to apply

    Args:
        df0 (pd.DataFrame): dataset with text columns
        input_col (str): input column
        output_col (str): output column

    Returns:
        pd.DataFrame: output dataset
    """
    df = df0.copy()
    df.loc[:, output_col] = df.loc[:, input_col].str.split(
        '___________________').str[1]

    df.loc[:, output_col] = df.loc[:, input_col].astype(
        str).apply(str_standard_cleaning)

    return df


def str_standard_cleaning(string: str) -> str:
    """Standard cleaning to apply

    Args:
        df0 (pd.DataFrame): dataset with text columns
        input_col (str): input column
        output_col (str): output column

    Returns:
        pd.DataFrame: output dataset
    """
    operations_ = [("(?<=[A-Za-z0-9])<br>(?=[A-Z])", ". "),
                   ("(?<=[0-9]) (?=[0-9])", ""),
                   ("(?<=[0-9a-z])\(", " ("),
                   ("[.]+", "."),
                   ("<br>", " "),
                   ("[_]+", "_"),
                   ("[ ]+", " "),
                   ("^[^A-Za-z0-9]+", "")]

    for i in operations_:
        string = re.sub(i[0], i[1], string)

    str_operations = [("\n", " "),
                      ("\t", " ")]
    for i in str_operations:
        string = string.replace(i[0], i[1])

    return string


def load_and_clean_cr_th(pwd, path: str, filename="CR_TH.csv", encoding="utf-8", delimiter=";"):
    """
    This function read csv files, removes double spaces and underscores. 
    Args:
        path(str): path to read csv file

    Returns:
        (pd.DataFrame, pd.DataFrame): first dataframe is cleaned. Second dataframe is raw.   
    """

    # if path is None:
    #    utl = utils_project()
    #    path = utl.params["path_local"] + utl.params["path_data"]
    zf = zipfile.ZipFile(path)
    df0 = pd.read_csv(zf.open(filename, 'r', pwd.encode()),
                      encoding=encoding, delimiter=delimiter)
    df = df0.copy()

    df = standard_cleaning(df, "contenu", "contenu")

    return df, df0


def read_text_from_protected_zipfile(path: str, pwd, zf, dc_correct=dc_correct) -> pd.Series:
    string = []

    with zf.open(path, 'r', pwd.encode()) as f:
        full_text = f.read().decode('iso8859-1')

        for item in list(dc_correct.items()):
            full_text = full_text.replace(item[0], item[1])
        full_text = ' '.join(full_text.lower().split()).encode(
            "utf-8").decode("utf-8")
        for l in re.split(r"(\.)", full_text):
            if l != ".":
                string.append(l)

    return pd.Series(string, name="contenu")


def load_trained_w2v_models():
    """Load trained w2v models.

    Returns:
        tuple: a tuple of three elements 
    """
    file_path = "./data/m_models/fr_w2v_fasttext.pkl"
    res_fasttext = pickle.load(open(file_path, 'rb'))
    file_path = "./data/m_models/fr_w2v.pkl"
    res_w2v = pickle.load(open(file_path, 'rb'))

    fast_text_embeddings = []
    words_vocab = list(res_w2v["vocabulary"].keys())
    for word_vocab in words_vocab:
        fast_text_embeddings.append(res_fasttext["w2v_model"].wv[word_vocab])
    fast_text_embeddings = np.array(fast_text_embeddings)
    fast_text_embeddings = pd.DataFrame(fast_text_embeddings)
    fast_text_embeddings.index = words_vocab

    return res_w2v, res_fasttext, fast_text_embeddings


def load_trained_glove_models():
    """Load trained GloVe models.

    Returns:
        tuple: a tuple of three elements 
    """
    file_path = "../../data/m_models/fr_w2v_fasttext.pkl"
    res_fasttext = pickle.load(open(file_path, 'rb'))
    file_path = "../../data/m_models/fr_w2v_glove.pkl"
    res_w2v = pickle.load(open(file_path, 'rb'))

    fast_text_embeddings = []
    words_vocab = list(res_w2v["vocabulary"].keys())
    for word_vocab in words_vocab:
        fast_text_embeddings.append(res_fasttext["w2v_model"].wv[word_vocab])
    fast_text_embeddings = np.array(fast_text_embeddings)
    fast_text_embeddings = pd.DataFrame(fast_text_embeddings)
    fast_text_embeddings.index = words_vocab

    return res_w2v, res_fasttext, fast_text_embeddings
