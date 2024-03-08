from HLab.hmd.text import get_window
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import dgl
from HLab.hmd import Utilities as Util
from HLab.hmd.preprocessing import *
from HLab.hmd.text import *


if __name__ == '__main__':
    data, labels = fetch_20newsgroups(
        data_home=Util.get_data_path('20newsgroups'),
        subset='all',
        return_X_y=True
    )

    edges_src = []
    edges_dst = []
    edge_features = []
    
    # 1. Preprocessing
    
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

    print(doc_nodes)
    print(word_nodes)
    
    # for idx, row in tqdm(enumerate(tfidf_vec), desc="generate tfidf edge"):
    #     for col_ind, value in zip(row.indices, row.data):
    #         edges_src.append(idx) # doc_id
    #         edges_dst.append(doc_nodes + col_ind) # word_id
    #         edge_features.append(value)

    # 3. PMI
    word_window_freq, word_pair_count, windows_count = get_window(corpus, 20)
    pmi_edge_lst = count_pmi(word_window_freq, word_pair_count, windows_count, threshold=0)

    i = 1
    for edge_item in pmi_edge_lst:
        w1_idx = doc_nodes + lexicon[edge_item[0]]
        w2_idx = doc_nodes + lexicon[edge_item[1]]

        print(edge_item)
        print(w1_idx)
        print(w2_idx)
        print(edge_item[0])
        print(lexicon[edge_item[0]])
        print(edge_item[1])
        print(lexicon[edge_item[2]])
        print(edge_item[2])
        if w1_idx == w2_idx:
            continue

        edges_src.append(w1_idx)
    
    # 4. Padding label for word node


    # 5. Create grpah

    # 6. Train test spit 

