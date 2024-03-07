import os
os.environ["DGLBACKEND"] = "pytorch"

import torch.nn.functional as F
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from classes import *
from HLab.hmd import Utilities as Util


def train(g, model):
    optimizer = th.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

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
    # dataset = CoraDataset()
    
    # g = dataset[0]
   
    
    # Util.save_graph('cora', g, g.ndata['label'], {'num_classes': dataset.num_classes})
    graph, labels, info = Util.load_graph('cora')
    model = GCN(graph.ndata["feat"].shape[1], 16, info['num_classes'])
    train(graph, model)
