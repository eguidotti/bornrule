import os
import re
import nltk
import tarfile
import zipfile
import warnings
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy import sparse
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
from torchvision import datasets, transforms
from pmlb import fetch_data

try:
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
except ModuleNotFoundError:
    warnings.warn("rdkit not installed: some (chemical) dataset may not work.")


class Dataset:

    def __init__(self, dataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.features_names = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        if hasattr(self, f"load_{dataset}"):
            getattr(self, f"load_{dataset}")()

        elif hasattr(datasets, dataset):
            self.X_train, self.y_train = self.torchvision(dataset, subset='train')
            self.X_test, self.y_test = self.torchvision(dataset, subset='test')

        else:
            df = fetch_data(dataset, local_cache_dir=self.output_dir)
            self.X_train, self.y_train = self.X_y(df, 'target')

        self.shape = self.X_train.shape[1:]
        self.ndim = len(self.shape)

        if self.ndim > 1:
            self.X_train = self.flatten(self.X_train)
            if self.X_test is not None:
                self.X_test = self.flatten(self.X_test)

    def torchvision(self, dataset, subset):
        loader = getattr(datasets, dataset)
        root = self.output_dir + "/" + dataset

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        try:
            data = loader(root=root, train=subset == 'train', download=True, transform=transform)

        except TypeError:
            data = loader(root=root, split=subset, download=True, transform=transform)

        X, y = [], []
        for i in tqdm(range(len(data))):
            X.append(data[i][0].numpy())
            y.append(data[i][1])

        return np.array(X), np.array(y)

    def load_20ng(self):
        train = fetch_20newsgroups(data_home=self.output_dir + "/20ng", subset='train')
        test = fetch_20newsgroups(data_home=self.output_dir + "/20ng", subset='test')
        
        vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=False)

        self.X_train = vectorizer.fit_transform(train.data)
        self.y_train = np.array([train.target_names[t] for t in train.target])

        self.X_test = vectorizer.transform(test.data)
        self.y_test = np.array([test.target_names[t] for t in test.target])

        self.features_names = vectorizer.get_feature_names_out()

    def load_hiv(self):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
        data_path = self.output_dir + "/hiv"

        if not os.path.exists(data_path):
            print("Downloading dataset (once and for all) into %s" % data_path)
            os.mkdir(data_path)
            urlretrieve(url, filename=data_path + "/hiv.csv")
            print("done.")

        df = pd.read_csv(data_path + "/hiv.csv")
        self.X_train, self.y_train = self.smiles_to_ecfp(df['smiles']), np.asarray(df['HIV_active'])

    def load_bace(self):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        data_path = self.output_dir + "/bace"

        if not os.path.exists(data_path):
            print("Downloading dataset (once and for all) into %s" % data_path)
            os.mkdir(data_path)
            urlretrieve(url, filename=data_path + "/bace.csv")
            print("done.")

        df = pd.read_csv(data_path + "/bace.csv")
        self.X_train, self.y_train = self.smiles_to_ecfp(df['mol']), np.asarray(df['Class'])

    def load_htru2(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip"
        archive = "HTRU2.zip"
        data_path = self.output_dir + "/htru2"

        if not os.path.exists(data_path):
            print("Downloading dataset (once and for all) into %s" % data_path)
            os.mkdir(data_path)
            archive_path = os.path.join(data_path, archive)
            urlretrieve(url, filename=archive_path)
            print("unzipping HTRU2 dataset...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            print("done.")

        df = pd.read_csv(self.output_dir + "/htru2/HTRU_2.csv", header=None)
        self.X_train = np.asarray(df.iloc[:, 0:-1])
        self.y_train = np.asarray(df.iloc[:, -1])

    def load_zoo(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
        labels = {1: 'Mammal', 2: 'Bird', 3: 'Reptile', 4: 'Fish', 5: 'Amphibian', 6: 'Bug', 7: 'Invertebrate'}
        columns = ['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed',
                   'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']
        df = pd.read_csv(url, header=None, names=columns).replace({'type': labels})

        vectorizer = OneHotEncoder()

        self.X_train = vectorizer.fit_transform(df.iloc[:, 1:-1])
        self.y_train = np.asarray(df.iloc[:, -1])

        self.features_names = vectorizer.get_feature_names_out()

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

        vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=True)

        self.X_train = vectorizer.fit_transform([doc[0] for doc in train])
        self.y_train = np.array([doc[1] for doc in train])

        self.X_test = vectorizer.transform([doc[0] for doc in test])
        self.y_test = np.array([doc[1] for doc in test])

        self.features_names = vectorizer.get_feature_names_out()

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

    def split(self, train_size=1, random_state=None):
        if self.X_test is not None and self.y_test is not None:
            n_samples = int(train_size * len(self.y_train))
            X_train, y_train = shuffle(self.X_train, self.y_train, n_samples=n_samples, random_state=random_state)
            X_test, y_test = self.X_test, self.y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_train, self.y_train, test_size=0.2, train_size=train_size * 0.8)

        return X_train, X_test, y_train, y_test

    def summary(self):
        n_train = self.X_train.shape[0]
        n_test = self.X_test.shape[0] if self.X_test is not None else None
        return {'n_train': n_train, 'n_test': n_test, 'shape': self.shape, 'classes': Counter(self.y_train)}

    @staticmethod
    def flatten(ndarray):
        return ndarray.reshape(ndarray.shape[0], -1)

    @staticmethod
    def X_y(df, target):
        categorical = []
        for i, dtype in df.dtypes.iteritems():
            if i == target:
                continue
            elif pd.api.types.is_integer_dtype(dtype):
                categorical.append(i)
            else:
                warnings.warn(f"Skipping feature '{i}'")

        return OneHotEncoder().fit_transform(df[categorical]), np.array(df[target])

    @staticmethod
    def smiles_to_ecfp(smiles):
        return sparse.csr_matrix(np.array([GetMorganFingerprintAsBitVect(MolFromSmiles(s), radius=2) for s in smiles]))
