from os.path import join as path_join
import numpy as np
import scipy.sparse as sp
import dgl
from dgl.data import DGLDataset
from HLab.hmd import Utilities as Util
import torch as th
from .hmd_helpers import *


class CoraDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="Cora")
        self.num_classes = 7
    
    def process(self):
        # <paper_id> <word_attributes>+ <class_label>
        idx_features_labels = np.genfromtxt(
            path_join(Util.get_data_path('cora'), 'cora.content'), 
            dtype=np.dtype(str)
        )

        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt(
            path_join(Util.get_data_path('cora'), 'cora.cites'), 
            dtype=np.int32
        )

        edges = np.array(
            list(
                map(
                    idx_map.get, 
                    edges_unordered.flatten()
                )
            ), dtype=np.int32).reshape(edges_unordered.shape)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        # coo_matrix((data, (i, j)), [shape=(M, N)])
        adj = sp.coo_matrix(
            (
                np.ones(edges.shape[0]), 
                (edges[:, 0], edges[:, 1])
            ),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32
        )

        # build symmetric adjacency matrix
        # đồ thị có hướng -> vô hướng 
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        g = dgl.from_scipy(adj)
        g = dgl.add_reverse_edges(g) # chuyển về đồ thị vô hướng
        self.g = dgl.add_self_loop(g) # + eye matrix

        features = normalize(features)

        self.g.ndata["feat"] = th.FloatTensor(np.array(features.todense()))
        self.g.ndata["label"] = th.LongTensor(np.where(labels)[1])

        n_nodes = g.num_nodes()
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = th.zeros(n_nodes, dtype=th.bool)
        val_mask = th.zeros(n_nodes, dtype=th.bool)
        test_mask = th.zeros(n_nodes, dtype=th.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.g.ndata["train_mask"] = train_mask
        self.g.ndata["val_mask"] = val_mask
        self.g.ndata["test_mask"] = test_mask
    

    def __getitem__(self, idx):
        return self.g


    def __len__(self):
        return 1
