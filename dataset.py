from typing import Dict, List, Generator
import pickle
import math

import torch
import torch.nn.functional as F


class Graph():
    # Adjacency matrix with the edge information: N x N x 1
    # `adj[i][j]` is `True` if there is an edge from `i` to `j`.
    adj: torch.Tensor
    # Initial features for each node: N x d
    features: torch.Tensor
    # Targets for each node: N
    targets: torch.Tensor

    def __init__(self, adj: torch.Tensor, features: torch.Tensor, targets: torch.Tensor):
        N = len(adj)
        d = features.size()[-1]
        assert adj.size() == (N,N)
        assert features.size() == (N,d)
        assert targets.size() == (N,)
        
        self.adj = torch.unsqueeze(adj, dim=-1)     # add extra third dimension, for the case there are multiple heads
        self.features = features
        self.targets = targets


class IdDataset():
    training_graphs: List[torch.Tensor]
    validation_graphs: List[torch.Tensor]
    test_graphs: List[torch.Tensor]

    def __init__(self, load_default=True):
        if load_default:
            self.load("data/IdDataset.pkl")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
            self.training_graphs = dataset.training_graphs
            self.validation_graphs = dataset.validation_graphs
            self.test_graphs = dataset.test_graphs

    def iter_batches(self, batch_size) -> Generator[List[Graph], None, None]:
        n_train = len(self.training_graphs)
        batch_idxs = torch.randperm(n_train)
        batch_idxs = torch.chunk(batch_idxs, math.ceil(n_train/batch_size))
        for batch in batch_idxs:
            yield [self.training_graphs[i] for i in batch]


def generateIdDataset() -> IdDataset:
    training_graphs = []
    validation_graphs = []
    test_graphs = []
    
    r = 0.2
    d = 5

    for n in range(20,30):
        for _ in range(15):
            training_graphs.append(
                generate_Id_graph(n,r,d)
            )
        for _ in range(5):
            validation_graphs.append(
                generate_Id_graph(n,r,d)
            )
        for _ in range(5):
            test_graphs.append(
                generate_Id_graph(n,r,d)
            )

    dataset = IdDataset(load_default=False)
    dataset.training_graphs = training_graphs
    dataset.validation_graphs = validation_graphs
    dataset.test_graphs = test_graphs
    return dataset


def generate_ER_adj(n,r):
    """
    generate a nxn undirected adjacency matrix according to the ER model,
    where the graph has n nodes and for any nodes u≠v,
    the probability that (u,v) is an edge is r
    """
    rand_nxn = torch.tril(torch.rand(size=(n,n)))
    # make the matrix symmetric by summing the lower triangular part with its transpose
    # further, add an identity matrix to make all entries on the diagonal >1
    rand_nxn_symmetric = torch.transpose(torch.tril(rand_nxn), 0, 1) + torch.tril(rand_nxn) + torch.eye(n)
    return rand_nxn_symmetric < r

def has_isolated_node(adj):
    return torch.any(torch.sum(adj, dim=0) == 0)

def generate_ER_adj_no_isolated(n,r):
    """
    generate a nxn adjacency matrix according to ER(n,r)
    while avoiding isolated nodes via rejection sampling
    """
    for i in range(1000):
        adj = generate_ER_adj(n,r)
        if not has_isolated_node(adj):
            return adj
        if i==20:
            print("20 failed attempts")
    raise Exception("Timeout")

def generate_random_feat_and_targ(n,d):
    """
    generate a nxd feature matrix 
    where each row is a uniformly chosen one-hot vector of size d
    an a d-size feature vector where each entry is the corresponding class index
    """
    targets = torch.randint(low=0, high=d, size=(n,))
    features = F.one_hot(targets, num_classes=d).float()
    return (features, targets)

def generate_Id_graph(n,r,d) -> Graph:
    adj = generate_ER_adj_no_isolated(n,r)
    feat, targ = generate_random_feat_and_targ(n,d)

    return Graph(
        adj=adj,
        features=feat,
        targets=targ
    )


if __name__ == "__main__":
    dataset = generateIdDataset()
    dataset.save("data/dataset.pkl")