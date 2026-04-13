import textwrap
import numpy as np
from numpy.linalg import norm


def split_long_strings(s: str, nb_char: int = 512) -> list:
    """Split string without splitting words.

    Args:
        s (str): String to split
        nb_char (_type_): Limit number of chars

    Returns:
        list: list of strings
    """
    s_split = []
    s = str(s).split(".")
    for i in s:
        s_split = s_split + textwrap.wrap(i, nb_char)

    return s_split


def calculate_similarity(x: np.array, y: np.array) -> float:
    """Calculate similarity between two arrays

    Args:
        x (np.array): embedding as an array
        y (np.array): embedding as an array

    Returns:
        float: cos similarity
    """
    return np.dot(x, y)/(norm(x)*norm(y))
