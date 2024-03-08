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
from scipy.sparse import coo_matrix


def normalize(adj):
    """ normalize adjacency matrix with normalization-trick that is faithful to
    the original paper.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # ~D in the GCN paper
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)



def normalize_pygcn(adj):
    """ normalize adjacency matrix with normalization-trick. This variant
    is proposed in https://github.com/tkipf/pygcn .
    Refer https://github.com/tkipf/pygcn/issues/11 for the author's comment.

    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix

    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    # no need to add identity matrix because self connection has already been added
    # a += sp.eye(a.shape[0])
    rowsum = np.array(adj.sum(1))
    rowsum_inv = np.power(rowsum, -1).flatten()
    rowsum_inv[np.isinf(rowsum_inv)] = 0.
    # ~D in the GCN paper
    d_tilde = sp.diags(rowsum_inv)
    return d_tilde.dot(adj)


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
    I = coo_matrix(np.eye(num_nodes))

    weighted_matrix = weighted_matrix + I
    



