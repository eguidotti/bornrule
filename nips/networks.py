import torch
import numpy as np
from scipy import sparse
from bornrule import BornClassifier
from bornrule.torch import Born


class BCBorn(torch.nn.Module):

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
        return self.born(torch.sqrt(x))


class SoftMax(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class CNN(torch.nn.Module):

    def __init__(self, shape):
        super().__init__()

        k, c = 5, 10
        self.conv = torch.nn.Conv2d(shape[0], out_channels=c, kernel_size=(k, k))

        p = 2
        self.pool = torch.nn.MaxPool2d(kernel_size=(p, p))

        self.out_features = c * (shape[1] - k + 1) // p * (shape[2] - k + 1) // p
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=shape)
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.unflatten(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class CNNBorn(torch.nn.Module):

    def __init__(self, shape, n_classes):
        super().__init__()
        self.cnn = CNN(shape=shape)
        self.born = Born(self.cnn.out_features, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.born(x)
        return x


class CNNSoftMax(torch.nn.Module):

    def __init__(self, shape, n_classes):
        super().__init__()
        self.cnn = CNN(shape=shape)
        self.softmax = SoftMax(in_features=self.cnn.out_features, out_features=n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.softmax(x)
        return x
