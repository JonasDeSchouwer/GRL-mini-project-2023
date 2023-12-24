import torch
import torch.nn as nn

from dataset import IdDataset
from experiment import evaluate


class IdentityModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        return x


def main():
    idmodel = IdentityModel(d=5)

    for s in range(0,10):
        dataset = IdDataset(load_default=False)
        dataset.load(f"study1/datasets/IdDataset_{s*10}°")

        eval_argmax = evaluate(idmodel, dataset.test_graphs)
        print(s*10, '°: ', eval_argmax["acc"], sep="")

if __name__ == "__main__":
    main()
