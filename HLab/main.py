from HLab.hmd.text import get_window
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
import dgl
from HLab.hmd import Utilities as Util
from HLab.hmd.preprocessing import *
from HLab.hmd.text import *
from sklearn.model_selection import train_test_split
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
import scipy.sparse as sp


if __name__ == '__main__':
    data, labels = fetch_20newsgroups(
        data_home=Util.get_data_path('20newsgroups'),
        subset='test',
        return_X_y=True
    )

    edges_src = []
    edges_dst = []
    edge_features = []

    preprocessor = StringPreprocessing()
    preprocessor.add_handler(ToLowerCase())
    preprocessor.add_handler(RemoveWhiteSpace())
    preprocessor.add_handler(RemovePunctuation())
    preprocessor.add_handler(EnglishTokenizer())

    corpus = [preprocessor.execute(d) for d in data]

    vectorizer = TfidfVectorizer(token_pattern=r"\S+")
    tfidf_vec = vectorizer.fit_transform(corpus)
    lexicon = vectorizer.vocabulary_

    doc_nodes = len(corpus)
    word_nodes = len(lexicon)
    num_nodes = doc_nodes + word_nodes

    for idx, row in tqdm(enumerate(tfidf_vec), desc="generate tfidf edge"):
        for col_ind, value in zip(row.indices, row.data):
            edges_src.append(idx) # doc_id
            edges_dst.append(doc_nodes + col_ind) # word_id
            edge_features.append(value)

    word_window_freq, word_pair_count, windows_count = get_window(corpus, 20)
    pmi_edge_lst = count_pmi(word_window_freq, word_pair_count, windows_count, threshold=0)

    for edge_item in pmi_edge_lst:    
        w1_idx = doc_nodes + lexicon[edge_item[0]]
        w2_idx = doc_nodes + lexicon[edge_item[1]]
        edges_src.append(w1_idx) # word_1
        edges_dst.append(w2_idx) # word_2
        edge_features.append(edge_item[2])

    labels = [lbl + 1 for lbl in labels]
    word_labels = [0] * word_nodes
    labels = labels + word_labels

    # edges_src = torch.from_numpy(np.array(edges_src))
    # edges_dst = torch.from_numpy(np.array(edges_dst))

    weighted_matrix = sp.csr_matrix(
        (edge_features, (np.array(edges_src), edges_dst)), 
        shape=(num_nodes, num_nodes)
    )
    print(weighted_matrix)

    features = torch.parse(np.array(weighted_matrix.todense()))
