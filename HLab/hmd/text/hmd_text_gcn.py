from collections import defaultdict
import itertools
from tqdm import tqdm
import math
from typing import Any


def get_window(corpus: list, window_size: int) -> tuple[defaultdict[Any, int], defaultdict[Any, int], int]:
    """ Calculate w(i), w(i, j) and count windows over the corpus

    Args:
        corpus (list) \n
        window_size (int)

    Returns:
        w(i), w(ij), w#
    """
    word_window_freq = defaultdict(int)  # w(i) 
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_count = 0

    for doc in tqdm(corpus, desc="Split by window"):
        windows = list()

        if isinstance(doc, str):
            words = doc.split()

        doc_length = len(words)

        if doc_length <= window_size:
            windows.append(words)
        else:
            for j in range(doc_length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_count += len(windows)
    return word_window_freq, word_pair_count, windows_count


def cal_pmi(wi_freq: int, wj_freq: int, wij_freq: int, windows_count: int) -> float:
    """ Calculate PMI score of a pair of word(wi, wj)

    Args:
        wi_freq (int) \n
        wj_freq (int) \n
        wij_freq (int) \n
        windows_count (int) \n

    Returns:
        PMI score
    """
    p_i = wi_freq / windows_count
    p_j = wj_freq / windows_count
    p_i_j = wij_freq / windows_count
    return math.log(p_i_j / (p_i * p_j))



def count_pmi(word_window_freq: int, word_pair_count: int, windows_count: int, threshold=0):
    word_pmi_lst = list()
    for word_pair, wij in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        wi_freq = word_window_freq[word_pair[0]]
        wj_freq = word_window_freq[word_pair[1]]

        pmi = cal_pmi(wi_freq, wj_freq, wij, windows_count)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst
