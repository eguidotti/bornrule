import torch
import numpy as np
from scipy import sparse
from bornrule import BornClassifier
from bornrule.torch import Born


class Quantum(torch.nn.Module):

    def __init__(self, in_features, out_features, Xy=None):
        super().__init__()
        self.born = Born(in_features=in_features, out_features=out_features)

        if Xy is not None:
            weight = BornClassifier().fit(Xy[0], Xy[1]).explain()
            weight = weight / np.mean(weight) / np.sqrt(in_features)
            if sparse.issparse(weight):
                weight = weight.todense()

            weight = torch.tensor(np.array([weight, np.zeros_like(weight)]), dtype=torch.get_default_dtype())
            self.born.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        x = torch.sqrt(x)
        x = self.born(x)
        return x
