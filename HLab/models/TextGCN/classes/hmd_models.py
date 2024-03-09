import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F


class TextGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(TextGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h
