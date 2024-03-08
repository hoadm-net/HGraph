import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
from sklearn.datasets import fetch_20newsgroups
from HLab.hmd import Utilities as Util
from HLab.hmd.preprocessing import *
from .hmd_helpers import *


class NGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='20 Newsgroups')

    def __getitem__(self, idx):
        return self.graph
    
    def __len__(self):
        return 1
    
    def process(self):
        data, labels = fetch_20newsgroups(
            data_home=Util.get_data_path('20newsgroups'),
            subset='all',
            return_X_y=True
        )
        
        # 1. Preprocessing
        
        preprocessor = StringPreprocessing()
        preprocessor.add_handler(ToLowerCase())
        preprocessor.add_handler(RemoveWhiteSpace())
        preprocessor.add_handler(RemovePunctuation())
        preprocessor.add_handler(VietnameseTokenizer())
        
        docs = [preprocessor.execute(d) for d in data]

        # 2. TF 2