import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from bornrule import BornClassifier
from .dataset import Dataset


class Experiment:

    def __init__(self, dataset, score, output_dir="results"):
        self.score = score
        self.scorer = make_scorer(self.score, greater_is_better=True)
        self.data = Dataset(dataset, output_dir=output_dir)

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.models = {
            'LR': (LogisticRegression(), {
                'solver': ['saga'],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.01, 0.1, 1, 10, 100],
                'fit_intercept': [True, False]
            }),
            'SVM': (SVC(), {
                'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto']
            }),
            'MNB': (MultinomialNB(), {
                'alpha': [0, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                'fit_prior': [True, False]
            }),
            'DT': (DecisionTreeClassifier(), {
                'splitter': ['best', 'random'],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 10, 100],
                'ccp_alpha': [0, 0.1, 1, 10]
            }),
            'RF': (RandomForestClassifier(), {
                'n_estimators': [10, 100, 1000],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 10, 100],
                'ccp_alpha': [0, 0.1, 1, 10]
            }),
            'KNN': (KNeighborsClassifier(), {
                'p': [1, 2],
                'n_neighbors': [1, 2, 3, 5, 10, 50, 100],
                'weights': ['uniform', 'distance']
            }),
            'BC': (BornClassifier(), {
                # no tuning
            })
        }

    def timing_cpu(self, runs=1):
        times = []
        file = self.output_dir + "/" + self.data.dataset + "_timing_cpu.csv"
        for run in range(runs):
            for train_size in (np.arange(10) + 1) / 10:
                X_train, X_test, y_train, y_test = self.data.split(train_size=train_size)

                for name, (model, params) in self.models.items():
                    print(f"Run {run + 1}/{runs}: executing {name} with train_size={train_size}")

                    fit_start = time()
                    model.fit(X_train, y_train)
                    fit_end = time()

                    predict_start = time()
                    y_pred = model.predict(X_test)
                    predict_end = time()

                    times.append({
                        'run': run+1,
                        'model': name,
                        'train_size': train_size,
                        'fit': fit_end - fit_start,
                        'predict': predict_end - predict_start,
                        'score': self.score(y_true=y_test, y_pred=y_pred)
                    })

                    print("writing to file", file)
                    pd.DataFrame(times).to_csv(file, index=False)

        return times

    def timing_gpu(self, runs=1):
        try:
            import cupy
            gpu = cupy.cuda.Device()

            def to_cupy(x):
                if sparse.issparse(x):
                    return cupy.sparse.csr_matrix(x)
                return cupy.array(x)

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "CuPy required but not installed. "
                "Please install CuPy at https://cupy.dev")

        times = []
        file = self.output_dir + "/" + self.data.dataset + "_timing_gpu.csv"
        for run in range(runs):
            for train_size in (np.arange(10) + 1) / 10:
                print(f"Run {run + 1}/{runs}: executing BC with train_size={train_size}")

                onehot = OneHotEncoder()
                X_train, X_test, y_train, y_test = self.data.split(train_size=train_size)
                X_train_gpu, X_test_gpu = to_cupy(X_train), to_cupy(X_test)
                y_train_gpu = to_cupy(onehot.fit_transform(y_train.reshape(-1, 1)).todense())

                for _ in ['warmup', 'run']:  # warmup GPU with first run
                    model = BornClassifier()

                    fit_start = time()
                    model.fit(X_train_gpu, y_train_gpu)
                    gpu.synchronize()
                    fit_end = time()

                    predict_start = time()
                    y_pred = model.predict(X_test_gpu)
                    gpu.synchronize()
                    predict_end = time()

                times.append({
                    'run': run + 1,
                    'model': "BC (GPU)",
                    'train_size': train_size,
                    'fit': fit_end - fit_start,
                    'predict': predict_end - predict_start,
                    'score': self.score(y_true=y_test, y_pred=[onehot.categories_[0][y] for y in y_pred.get()])
                })

                print("writing to file", file)
                pd.DataFrame(times).to_csv(file, index=False)

        return times

    def cross_validation(self, runs=1):
        scores = []
        file = self.output_dir + "/" + self.data.dataset + "_cross_validation.csv"
        for run in range(runs):
            X_train, X_test, y_train, y_test = self.data.split()

            for name, (model, parameters) in self.models.items():
                print(f"Run {run + 1}/{runs}: executing {name}")
                clf = GridSearchCV(model, parameters, scoring=self.scorer, n_jobs=-1, verbose=2) if parameters is not None else model

                fit_start = time()
                clf.fit(X_train, y_train)
                fit_end = time()

                predict_start = time()
                y_pred = clf.predict(X_test)
                predict_end = time()

                scores.append({
                    'run': run + 1,
                    'model': name,
                    'fit': fit_end - fit_start,
                    'predict': predict_end - predict_start,
                    'score': self.score(y_true=y_test, y_pred=y_pred)
                })

                print("Writing to file", file)
                pd.DataFrame(scores).to_csv(file, index=False)

        return scores

    def plot_timing(self):
        timing = []
        for device in ['cpu', 'gpu']:
            file = f"{self.output_dir}/{self.data.dataset}_timing_{device}.csv"
            if os.path.exists(file):
                timing.append(pd.read_csv(file))

        df = pd.concat(timing).groupby(['model', 'train_size']).describe().reset_index()
        df.columns = [' '.join(col).strip() for col in df.columns]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        plt.tight_layout(pad=3, rect=(0, 0, 1, 0.95))

        for key, group in df.groupby('model'):
            args = {'label': key, 'legend': False, 'marker': ".", 'capsize': 2, 'elinewidth': 1}
            if key.startswith("BC"):
                args['color'] = 'black'
            if key == "BC (GPU)":
                args['linestyle'] = 'dotted'
                args['marker'] = 'x'
            group.plot(x='train_size', y='fit mean', yerr='fit std', ax=ax1, **args)
            group.plot(x='train_size', y='predict mean', yerr='predict std', ax=ax2, **args)
            group.plot(x='train_size', y='score mean', yerr='score std', ax=ax3, **args)

        for ax, label in [(ax1, "Training Time (s)"), (ax2, "Prediction Time (s)"), (ax3, "Accuracy Score")]:
            ax.set_xlabel('Dataset Size\n(fraction of training data)', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.setp(ax.spines.values(), linewidth=1.5)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

        for ax in [ax1, ax2]:
            ax.set_yscale('log')

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(df.model.unique()), prop={"size": 12})

        fig.savefig(f"{self.output_dir}/{self.data.dataset}_timing.png", bbox_inches='tight', format='png', dpi=300)
