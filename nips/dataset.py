import os
import re
import nltk
import tarfile
import torch
import numpy as np
import pandas as pd
from glob import glob
from scipy import sparse
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class Dataset:

    def __init__(self, dataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir

        self.vectorizer = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        getattr(self, f"load_{dataset}")()

    def load_20ng(self):
        train = fetch_20newsgroups(subset='train')
        test = fetch_20newsgroups(subset='test')
        
        self.vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=False)
        self.X_train = self.vectorizer.fit_transform(train.data)
        self.X_test = self.vectorizer.transform(test.data)
        self.y_train = np.array([train.target_names[t] for t in train.target])
        self.y_test = np.array([test.target_names[t] for t in test.target])

    def load_zoo(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
        labels = {1: 'Mammal', 2: 'Bird', 3: 'Reptile', 4: 'Fish', 5: 'Amphibian', 6: 'Bug', 7: 'Invertebrate'}
        columns = ['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed',
                   'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']
        df = pd.read_csv(url, header=None, names=columns).replace({'type': labels})

        self.vectorizer = OneHotEncoder()
        self.X_train = self.vectorizer.fit_transform(df.iloc[:, 1:-1])
        self.y_train = np.asarray(df.iloc[:, -1])

    def load_r8(self):
        self.load_reuters('r8')

    def load_r52(self):
        self.load_reuters('r52')

    def load_reuters(self, sample):
        train, test = self.download_reuters()
        if sample == 'r52':
            labels = {doc[1] for doc in train}.intersection({doc[1] for doc in test})
        elif sample == 'r8':
            labels = ['grain', 'earn', 'interest', 'acq', 'trade', 'crude', 'ship', 'money-fx']
        else:
            raise ValueError("sample must be one of: 'r8' or 'r52'")

        train = [doc for doc in train if doc[1] in labels]
        test = [doc for doc in test if doc[1] in labels]

        self.vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=True)
        self.X_train = self.vectorizer.fit_transform([doc[0] for doc in train])
        self.X_test = self.vectorizer.transform([doc[0] for doc in test])
        self.y_train = np.array([doc[1] for doc in train])
        self.y_test = np.array([doc[1] for doc in test])

    def download_reuters(self):
        train, test = [], []
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
        archive = "reuters21578.tar.gz"
        data_path = self.output_dir + "/reuters"

        if not os.path.exists(data_path):
            print("Downloading dataset (once and for all) into %s" % data_path)
            os.mkdir(data_path)
            archive_path = os.path.join(data_path, archive)
            urlretrieve(url, filename=archive_path)
            print("untarring Reuters dataset...")
            tarfile.open(archive_path, "r:gz").extractall(data_path)
            print("done.")

        for filename in glob(os.path.join(data_path, "*.sgm")):
            with open(filename, encoding='ISO-8859-1') as f:
                for node in BeautifulSoup(f.read(), 'html.parser').find_all('reuters'):
                    text = re.sub(r'\s+', ' ', node.find('text').text)
                    labels = [n.text for n in node.topics.find_all('d')]
                    if len(labels) == 1 and node['topics'] == "YES":
                        if node['lewissplit'] == 'TRAIN':
                            train.append((text, labels[0]))
                        elif node['lewissplit'] == 'TEST':
                            test.append((text, labels[0]))

        return train, test

    def summary(self):
        n_classes = len(np.unique(self.y_train))
        n_train, n_features = self.X_train.shape
        n_test = self.X_test.shape[0] if self.X_test is not None else None
        return {'n_train': n_train, 'n_test': n_test, 'n_features': n_features, 'n_classes': n_classes}

    def split(self, train_size=1):
        if self.X_test is not None and self.y_test is not None:
            X_train, y_train = shuffle(self.X_train, self.y_train, n_samples=int(train_size * len(self.y_train)))
            X_test, y_test = self.X_test, self.y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_train, self.y_train, test_size=0.2, train_size=train_size * 0.8)

        if sparse.issparse(X_train):
            columns = np.unique(sparse.find(X_train)[1])
        else:
            columns = ~np.all(X_train == 0, axis=0)

        return X_train[:, columns], X_test[:, columns], y_train, y_test

    def to_torch(self, X_train, X_test, y_train, y_test, batch_size=128, device='cpu'):
        ohe = OneHotEncoder()
        X_train_nn = self.to_tensor(X_train, device=device)
        X_test_nn = self.to_tensor(X_test, device=device)
        y_train_nn = self.to_tensor(ohe.fit_transform(y_train.reshape(-1, 1)), device=device).to_dense()
        y_test_nn = self.to_tensor(ohe.transform(y_test.reshape(-1, 1)), device=device).to_dense()
        train_loader = DataLoader(TensorDataset(X_train_nn, y_train_nn), batch_size=batch_size, shuffle=True)
        test_data = (X_test_nn, y_test_nn)
        return train_loader, test_data

    @staticmethod
    def to_tensor(x, device):
        if sparse.issparse(x):
            x = x.tocoo()
            i = torch.LongTensor(np.vstack((x.row, x.col)))
            v = torch.tensor(x.data, device=device)
            return torch.sparse_coo_tensor(i, v, torch.Size(x.shape))
        return torch.tensor(x, device=device)
