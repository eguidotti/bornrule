import torch
import numpy as np
from scipy import sparse
from bornrule import BornClassifier
from bornrule.torch import Born


class SoftMax(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BCBorn(Born):

    def __init__(self, X, y):
        in_features, out_features = X.shape[1], len(np.unique(y))

        weight = BornClassifier().fit(X, y).explain()
        weight = weight / np.mean(weight) / np.sqrt(in_features)
        if sparse.issparse(weight):
            weight = weight.todense()

        super().__init__(in_features=in_features, out_features=out_features)
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, x):
        return super().forward(torch.sqrt(x))
