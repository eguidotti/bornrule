import os
import re
import tarfile
import numpy as np
from glob import glob
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


class Dataset:

    def __init__(self, dataset, output_dir):
        self.dataset = dataset

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.features_names = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        getattr(self, f"load_{dataset}")()

    def bag_of_words(self, x):
        bow = []
        for i in range(x.shape[0]):
            item = x[i].tocoo()
            data = item.data
            keys = [self.features_names[c] for c in item.col]
            bow.append(dict(zip(keys, data)))
        return bow

    def load_20ng(self):
        train = fetch_20newsgroups(data_home=self.output_dir + "/20ng", subset='train')
        test = fetch_20newsgroups(data_home=self.output_dir + "/20ng", subset='test')
        
        vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=False, token_pattern=None)

        self.X_train = vectorizer.fit_transform(train.data)
        self.y_train = np.array([train.target_names[t] for t in train.target])

        self.X_test = vectorizer.transform(test.data)
        self.y_test = np.array([test.target_names[t] for t in test.target])

        self.features_names = vectorizer.get_feature_names_out()
        self.X_train = self.bag_of_words(self.X_train)
        self.X_test = self.bag_of_words(self.X_test)

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

        vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=True, token_pattern=None)

        self.X_train = vectorizer.fit_transform([doc[0] for doc in train])
        self.y_train = np.array([doc[1] for doc in train])

        self.X_test = vectorizer.transform([doc[0] for doc in test])
        self.y_test = np.array([doc[1] for doc in test])

        self.features_names = vectorizer.get_feature_names_out()
        self.X_train = self.bag_of_words(self.X_train)
        self.X_test = self.bag_of_words(self.X_test)

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
