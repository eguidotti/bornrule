import torch
import numpy as np
from scipy import sparse
from bornrule import BornClassifier
from bornrule.torch import Born


class BCBorn(torch.nn.Module):

    def __init__(self, X, y, dtype):
        super().__init__()
        in_features, out_features = X.shape[1], len(np.unique(y))

        weight = BornClassifier().fit(X, y).explain()
        weight = weight / np.mean(weight) / np.sqrt(in_features)
        if sparse.issparse(weight):
            weight = weight.todense()

        self.born = Born(in_features=in_features, out_features=out_features)
        self.born.weight = torch.nn.Parameter(torch.tensor(weight, dtype=dtype))

    def forward(self, x):
        x = torch.sqrt(x)
        x = self.born(x)
        return x


class SoftMax(torch.nn.Module):

    def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, dtype=dtype)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x
