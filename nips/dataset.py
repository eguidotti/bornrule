import nltk
from scipy import sparse
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, dataset):
        self.dataset = dataset
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        getattr(self, f"load_{dataset}")()

    def load_20ng(self):
        train = fetch_20newsgroups(subset='train')
        test = fetch_20newsgroups(subset='test')
        vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=False)
        self.X_train = vectorizer.fit_transform(train.data)
        self.X_test = vectorizer.transform(test.data)
        self.y_train, self.y_test = train.target, test.target

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
