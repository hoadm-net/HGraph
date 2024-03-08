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
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h



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


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["features"]
    labels = g.ndata["labels"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(200):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)
        print(pred)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )

if __name__ == '__main__':
    data, labels = fetch_20newsgroups(
        data_home=Util.get_data_path('20newsgroups'),
        subset='all',
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

    edges_src = torch.from_numpy(np.array(edges_src))
    edges_dst = torch.from_numpy(np.array(edges_dst))

    graph = dgl.graph(
        (edges_src, edges_dst), num_nodes=num_nodes
    )

    graph = dgl.add_reverse_edges(graph) # chuyển về đồ thị vô hướng
    graph = dgl.add_self_loop(graph) # + eye matrix
    graph.ndata["labels"] = torch.from_numpy(np.array(labels))

    weighted_matrix = sp.coo_matrix(
        (edge_features, (np.array(edges_src), edges_dst)), 
        shape=(num_nodes, num_nodes)
    )
    I = sp.identity(num_nodes, format='coo')

    weighted_matrix = weighted_matrix + I
    weighted_matrix = weighted_matrix
    adj = normalize(weighted_matrix)
    adj = adj.tocoo()
    
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    graph.ndata["features"] = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense()
 
    node_indices = [i for i in range(num_nodes)]

    x_train, x_test, y_train, y_test = train_test_split(node_indices, labels, test_size=0.33, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=42)    

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[x_train] = True
    val_mask[val_mask] = True
    test_mask[test_mask] = True

    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask

    num_classes = len(set(labels)) + 1
    model = GCN(graph.ndata["features"].shape[1], 200, num_classes)
    train(graph, model)
    