import os

os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel
from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GINConv, SumPooling
import torch.multiprocessing as mp
from dgl.data import GINDataset


class GIN(nn.Module):
    def __init__(self, input_size=1, num_classes=2):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            nn.Linear(input_size, num_classes), aggregator_type="sum"
        )
        self.conv2 = GINConv(
            nn.Linear(num_classes, num_classes), aggregator_type="sum"
        )
        self.pool = SumPooling()

    def forward(self, g, feats):
        feats = self.conv1(g, feats)
        feats = F.relu(feats)
        feats = self.conv2(g, feats)

        return self.pool(g, feats)


def init_process_group(world_size, rank):
    dist.init_process_group(
        backend="gloo",  # change to 'nccl' for multiple GPUs
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )


def get_dataloaders(dataset, seed, batch_size=32):
    # Use a 80:10:10 train-val-test split
    train_set, val_set, test_set = split_dataset(
        dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed
    )
    train_loader = GraphDataLoader(
        train_set, use_ddp=True, batch_size=batch_size, shuffle=True
    )
    val_loader = GraphDataLoader(val_set, batch_size=batch_size)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def init_model(seed, device):
    torch.manual_seed(seed)
    model = GIN().to(device)
    if device.type == "cpu":
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    return model


def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    total_correct = 0

    for bg, labels in dataloader:
        bg = bg.to(device)
        labels = labels.to(device)
        # Get input node features
        feats = bg.ndata.pop("attr")
        with torch.no_grad():
            pred = model(bg, feats)
        _, pred = torch.max(pred, 1)
        total += len(labels)
        total_correct += (pred == labels).sum().cpu().item()

    return 1.0 * total_correct / total


def run(rank, world_size, dataset, seed=0):
    init_process_group(world_size, rank)
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model = init_model(seed, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    train_loader, val_loader, test_loader = get_dataloaders(dataset, seed)
    for epoch in range(5):
        model.train()
        # The line below ensures all processes use a different
        # random ordering in data loading for each epoch.
        train_loader.set_epoch(epoch)

        total_loss = 0
        for bg, labels in train_loader:
            bg = bg.to(device)
            labels = labels.to(device)
            feats = bg.ndata.pop("attr")
            pred = model(bg, feats)

            loss = criterion(pred, labels)
            total_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = total_loss
        print("Loss: {:.4f}".format(loss))

        val_acc = evaluate(model, val_loader, device)
        print("Val acc: {:.4f}".format(val_acc))

    test_acc = evaluate(model, test_loader, device)
    print("Test acc: {:.4f}".format(test_acc))
    dist.destroy_process_group()


def main():
    if not torch.cuda.is_available():
        print("No GPU found!")
        return

    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    dataset = GINDataset(name="IMDBBINARY", self_loop=False)
    mp.spawn(run, args=(num_gpus, dataset), nprocs=num_gpus)


if __name__ == "__main__":
    main()
