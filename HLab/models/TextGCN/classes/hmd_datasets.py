from os.path import join as path_join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl.data import DGLDataset
from HLab.hmd import Utilities as Util
from HLab.hmd.preprocessing import *
from HLab.hmd.text import *


class NGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='20 Newsgroups')

    def __getitem__(self, idx):
        return self.g
    
    def __len__(self):
        return 1
    
    def process(self):
        
        edges_src = []
        edges_dst = []
        edge_features = []
        doc_nodes, word_nodes, num_nodes = 0, 0, 0

        data, labels = fetch_20newsgroups(
            data_home=Util.get_data_path('20newsgroups'),
            subset='all',
            return_X_y=True
        )

        self.num_classes = len(set(labels)) + 1

        # 1 - Preprocessing
        preprocessor = StringPreprocessing()
        preprocessor.add_handler(ToLowerCase())
        preprocessor.add_handler(RemoveWhiteSpace())
        preprocessor.add_handler(RemovePunctuation())
        preprocessor.add_handler(EnglishTokenizer())
        
        corpus = [preprocessor.execute(d) for d in data]
        
        # 2. TF-IDF
        vectorizer = TfidfVectorizer(token_pattern=r"\S+")
        tfidf_vec = vectorizer.fit_transform(corpus)
        lexicon = vectorizer.vocabulary_

        doc_nodes = len(corpus)
        word_nodes = len(lexicon)
        num_nodes = doc_nodes + word_nodes

        for idx, row in tqdm(enumerate(tfidf_vec), desc="Generate TF-IDF Edges"):
            for col_ind, value in zip(row.indices, row.data):
                edges_src.append(idx) # doc_id
                edges_dst.append(doc_nodes + col_ind) # word_id
                edge_features.append(value)

        # 3. PMI
        word_window_freq, word_pair_count, windows_count = get_window(corpus, 20)
        pmi_edge_lst = count_pmi(word_window_freq, word_pair_count, windows_count, threshold=0)
        for edge_item in pmi_edge_lst:    
            w1_idx = doc_nodes + lexicon[edge_item[0]]
            w2_idx = doc_nodes + lexicon[edge_item[1]]
            edges_src.append(w1_idx) # word_1
            edges_dst.append(w2_idx) # word_2
            edge_features.append(edge_item[2])
    
        # 4. Create Graph
        g = dgl.graph(
            (torch.tensor(edges_src), torch.tensor(edges_dst))
        )

        g = dgl.add_reverse_edges(g) # chuyển về đồ thị vô hướng
        self.g = dgl.add_self_loop(g) # + I

        # 5. features and labels
        features = sp.coo_matrix(
            (edge_features, (edges_src, edges_dst)), 
            shape=(num_nodes, num_nodes)
        )

        labels = [lbl + 1 for lbl in labels]
        word_labels = [0] * word_nodes
        labels = labels + word_labels

        values = features.data
        indices = np.vstack((features.row, features.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = features.shape

        self.g.ndata["features"] = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        self.g.ndata["label"] = torch.LongTensor(labels)

        # 6. train_test_split
        node_indices = [i for i in range(doc_nodes)]

        x_train, x_test, y_train, y_test = train_test_split(node_indices, node_indices, test_size=0.33, random_state=42, shuffle=False)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=42, shuffle=False)    

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[x_train] = True
        val_mask[x_val] = True
        test_mask[x_test] = True

        self.g.ndata['train_mask'] = train_mask
        self.g.ndata['val_mask'] = val_mask
        self.g.ndata['test_mask'] = test_mask


class UIT_VSFCDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='UIT VSFC')

    def __getitem__(self, idx):
        return self.g
    
    def __len__(self):
        return 1
    
    def process(self):
        data_path = Util.get_data_path('UIT-VSFC')
        train_path = path_join(data_path, 'train')
        test_path = path_join(data_path, 'test')
        val_path = path_join(data_path, 'dev')

        train_data = read_all_text(path_join(train_path, 'sents.txt'))
        train_labels = read_all_text(path_join(train_path, 'topics.txt'))

        test_data = read_all_text(path_join(test_path, 'sents.txt'))
        test_labels = read_all_text(path_join(test_path, 'topics.txt'))

        val_data = read_all_text(path_join(val_path, 'sents.txt'))
        val_labels = read_all_text(path_join(val_path, 'topics.txt'))

        train_len = len(train_data)
        test_len = len(test_data)
        val_len = len(val_data)

        data = train_data + test_data + val_data
        labels = train_labels + test_labels + val_labels
        labels = [int(l) for l in labels]

        edges_src = []
        edges_dst = []
        edge_features = []
        doc_nodes, word_nodes, num_nodes = 0, 0, 0

        self.num_classes = len(set(labels)) + 1

        # 1 - Preprocessing
        preprocessor = StringPreprocessing()
        preprocessor.add_handler(ToLowerCase())
        preprocessor.add_handler(RemoveWhiteSpace())
        preprocessor.add_handler(RemovePunctuation())
        preprocessor.add_handler(VietnameseTokenizer())
        
        corpus = [preprocessor.execute(d) for d in data]

        # 2. TF-IDF
        vectorizer = TfidfVectorizer(token_pattern=r"\S+")
        tfidf_vec = vectorizer.fit_transform(corpus)
        lexicon = vectorizer.vocabulary_

        doc_nodes = len(corpus)
        word_nodes = len(lexicon)
        num_nodes = doc_nodes + word_nodes

        for idx, row in tqdm(enumerate(tfidf_vec), desc="Generate TF-IDF Edges"):
            for col_ind, value in zip(row.indices, row.data):
                edges_src.append(idx) # doc_id
                edges_dst.append(doc_nodes + col_ind) # word_id
                edge_features.append(value)

        # 3. PMI
        word_window_freq, word_pair_count, windows_count = get_window(corpus, 20)
        pmi_edge_lst = count_pmi(word_window_freq, word_pair_count, windows_count, threshold=0)
        for edge_item in pmi_edge_lst:    
            w1_idx = doc_nodes + lexicon[edge_item[0]]
            w2_idx = doc_nodes + lexicon[edge_item[1]]
            edges_src.append(w1_idx) # word_1
            edges_dst.append(w2_idx) # word_2
            edge_features.append(edge_item[2])

        # 4. Create Graph
        g = dgl.graph(
            (torch.tensor(edges_src), torch.tensor(edges_dst))
        )

        g = dgl.add_reverse_edges(g) # chuyển về đồ thị vô hướng
        self.g = dgl.add_self_loop(g) # + I

        # 5. features and labels
        features = sp.coo_matrix(
            (edge_features, (edges_src, edges_dst)), 
            shape=(num_nodes, num_nodes)
        )

        labels = [lbl + 1 for lbl in labels]
        word_labels = [0] * word_nodes
        labels = labels + word_labels

        values = features.data
        indices = np.vstack((features.row, features.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = features.shape

        self.g.ndata["features"] = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        self.g.ndata["label"] = torch.LongTensor(labels)

        # 6. train_test_split
        x_train = [i for i in range(train_len)]
        x_test = [i for i in range(train_len, train_len + test_len)]
        x_val = [i for i in range(train_len + test_len, train_len + test_len + val_len)]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[x_train] = True
        val_mask[x_val] = True
        test_mask[x_test] = True

        self.g.ndata['train_mask'] = train_mask
        self.g.ndata['val_mask'] = val_mask
        self.g.ndata['test_mask'] = test_mask
