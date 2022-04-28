import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from scipy import sparse
from sklearn import metrics
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
from .networks import Quantum


class Experiment:

    def __init__(self, dataset, loss, score, needs_proba=False, device='cpu', data_dir="data", output_dir="results"):
        self.loss = loss
        self.score, self.needs_proba = score, needs_proba
        self.scorer = metrics.make_scorer(self.score, greater_is_better=True, needs_proba=needs_proba)
        self.data = Dataset(dataset, output_dir=data_dir)
        self.device = torch.device(device)

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
            'SVM': (SVC(probability=self.needs_proba), {
                'probability': [self.needs_proba],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto']
            }),
            'MNB': (MultinomialNB(), {
                'alpha': [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
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
                    y_pred = model.predict_proba(X_test) if self.needs_proba else model.predict(X_test)
                    predict_end = time()

                    if self.needs_proba and y_pred.shape[1] == 2:
                        y_pred = y_pred[:, 1]

                    times.append({
                        'run': run+1,
                        'model': name,
                        'train_size': train_size,
                        'fit': fit_end - fit_start,
                        'predict': predict_end - predict_start,
                        'score': self.score(y_test, y_pred)
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
                    y_pred_gpu = model.predict_proba(X_test_gpu) if self.needs_proba else model.predict(X_test_gpu)
                    gpu.synchronize()
                    predict_end = time()

                y_pred = y_pred_gpu.get()
                if not self.needs_proba:
                    y_pred = [onehot.categories_[0][y] for y in y_pred]
                elif y_pred.shape[1] == 2:
                    y_pred = y_pred[:, 1]

                times.append({
                    'run': run + 1,
                    'model': "BC (GPU)",
                    'train_size': train_size,
                    'fit': fit_end - fit_start,
                    'predict': predict_end - predict_start,
                    'score': self.score(y_test, y_pred)
                })

                print("writing to file", file)
                pd.DataFrame(times).to_csv(file, index=False)

        return times

    def plot_timing(self, score_label='Score'):
        timing = []
        for device in ['cpu', 'gpu']:
            file = f"{self.output_dir}/{self.data.dataset}_timing_{device}.csv"
            if os.path.exists(file):
                timing.append(pd.read_csv(file))

        df = pd.concat(timing).groupby(['model', 'train_size']).describe().reset_index()
        df.columns = [' '.join(col).strip() for col in df.columns]
        for column in ['fit', 'predict', 'score']:
            df[f"{column} err"] = df[f"{column} std"] / np.sqrt(df["run count"])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.3))
        plt.tight_layout(pad=3, rect=(0, 0, 1, 0.95))

        for key, group in df.groupby('model'):
            args = {'label': key, 'legend': False, 'marker': ".", 'capsize': 2, 'linewidth': 1, 'elinewidth': 0.5}
            if key.startswith("BC"):
                args['color'] = 'black'
            if key == "BC (GPU)":
                args['linestyle'] = 'dotted'
                args['marker'] = 'x'
            group.plot(x='train_size', y='fit mean', yerr='fit err', ax=ax1, **args)
            group.plot(x='train_size', y='predict mean', yerr='predict err', ax=ax2, **args)
            group.plot(x='train_size', y='score mean', yerr='score err', ax=ax3, **args)

        for ax, label in [(ax1, "Training Time (s)"), (ax2, "Prediction Time (s)"), (ax3, score_label)]:
            ax.set_xlabel('Dataset Size\n(fraction of training data)', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.setp(ax.spines.values(), linewidth=0.5)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

        for ax in [ax1, ax2]:
            ax.set_yscale('log')

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(df.model.unique()), prop={"size": 12})

        file = f"{self.output_dir}/{self.data.dataset}_timing.png"
        fig.savefig(file, bbox_inches='tight', format='png', dpi=300)
        print(f"Image saved in {file}")

    def cross_validation(self, runs=1):
        scores = []
        file = self.output_dir + "/" + self.data.dataset + "_cross_validation.csv"
        for run in range(runs):
            X_train, X_test, y_train, y_test = self.data.split()

            for name, (model, parameters) in self.models.items():
                if parameters:
                    print(f"Run {run + 1}/{runs}: executing {name}")
                    clf = GridSearchCV(model, parameters, scoring=self.scorer, verbose=3)

                    fit_start = time()
                    clf.fit(X_train, y_train)
                    fit_end = time()

                    predict_start = time()
                    y_pred = clf.predict_proba(X_test) if self.needs_proba else clf.predict(X_test)
                    predict_end = time()

                    if self.needs_proba and y_pred.shape[1] == 2:
                        y_pred = y_pred[:, 1]

                    scores.append({
                        'run': run + 1,
                        'model': name,
                        'fit': fit_end - fit_start,
                        'predict': predict_end - predict_start,
                        'score': self.score(y_test, y_pred)
                    })

                    print("Writing to file", file)
                    pd.DataFrame(scores).to_csv(file, index=False)

        return scores

    def ablation_study(self, runs=1):
        model = BornClassifier()
        parameters = {
            'a': [0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            'b': [0.00, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            'h': [0.00, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        }

        scores = []
        file = self.output_dir + "/" + self.data.dataset + "_ablation.csv"
        for run in range(runs):
            X_train, X_test, y_train, y_test = self.data.split()
            clf = GridSearchCV(model, parameters, scoring=self.scorer, n_jobs=-1, verbose=3)
            clf.fit(X_train, y_train)
            for params, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
                model.set_params(**params)
                model.fit(X_train, y_train)

                scores.append({
                    'run': run,
                    'a': params['a'],
                    'b': params['b'],
                    'h': params['h'],
                    'score_validation': score,
                    'score_test': self.scorer(estimator=model, X=X_test, y_true=y_test)
                })

            print("Writing to file", file)
            pd.DataFrame(scores).to_csv(file, index=False)

        return scores

    def plot_ablation(self):
        file = f"{self.output_dir}/{self.data.dataset}_ablation.csv"
        df = pd.read_csv(file).groupby(['a', 'b', 'h']).mean().reset_index()

        df['size'] = 20
        df.loc[(df['a'] == 0.5) & (df['b'] == 1) & (df['h'] == 1), 'size'] = 200

        x, y, z = 'b', 'a', 'h'
        c, s = 'score_test', 'size'

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(x), ax.set_ylabel(y), ax.set_zlabel(z)
        args = {'cmap': 'ocean', 'edgecolors': 'black', 'linewidths': 0.1, 'depthshade': 0}
        plot = ax.scatter(df[x], df[y], df[z], c=df[c], s=df[s], **args)
        fig.colorbar(plot, location='left')

        file = f"{self.output_dir}/{self.data.dataset}_ablation.png"
        fig.savefig(file, bbox_inches='tight', format='png', dpi=300)
        print(f"Image saved in {file}")

        s = df.iloc[df['score_validation'].idxmax()]
        print(f"Model: best cross validation. Accuracy on test set: {s['score_test']}")

        s = df.loc[(df['a'] == 0.5) & (df['b'] == 1) & (df['h'] == 1)].iloc[0]
        print(f"Model: Born Classifier. Accuracy on test set: {s['score_test']}")

        s = df.loc[(df['a'] == 0.5) & (df['b'] == 1) & (df['h'] == 0)].iloc[0]
        print(f"Model: Born Classifier without entropy. Accuracy on test set: {s['score_test']}")

        s = df.loc[(df['a'] == 1) & (df['b'] == 0) & (df['h'] == 0)].iloc[0]
        print(f"Model: Bayes Classifier. Accuracy on test set: {s['score_test']}")

        s = df.loc[(df['a'] == 1) & (df['b'] == 0) & (df['h'] == 1)].iloc[0]
        print(f"Model: Bayes Classifier with entropy. Accuracy on test set: {s['score_test']}")

    def table_explanation(self, top=10):
        file = self.output_dir + "/" + self.data.dataset + "_explanation.csv"

        model = BornClassifier()
        weights = model.fit(self.data.X_train, self.data.y_train).explain()
        classes, features = model.classes_, self.data.features_names

        if sparse.issparse(weights):
            df = pd.DataFrame.sparse.from_spmatrix(weights, index=features, columns=classes)
        else:
            df = pd.DataFrame(weights, index=features, columns=classes)

        print("Writing to file", file)
        top10 = pd.DataFrame({c: df[c].sort_values(ascending=False).index[0:top] for c in df.columns})
        top10.to_csv(file, index=True)

        return top10

    def learning_curve(self, epochs=1, runs=1, batch_size=128):
        scores = []
        file = self.output_dir + "/" + self.data.dataset + "_learning_curve.csv"
        for run in range(runs):
            X_train, X_test, y_train, y_test = self.data.split()
            in_features, out_features = X_train.shape[1], len(np.unique(y_train))
            train_batches, test_data = self.to_torch(X_train, X_test, y_train, y_test, batch_size)

            nets = {
                'BC': Quantum(in_features, out_features, (X_train, y_train), device=self.device),
                'BC+BL': Quantum(in_features, out_features, (X_train, y_train), device=self.device),
                'BL': Quantum(in_features, out_features, device=self.device),
            }

            for name, net in nets.items():
                print(f"--- Run: {run + 1}/{runs} ({name}) ---")

                score = self.train_and_eval(
                    net=net,
                    train_batches=train_batches,
                    test_data=test_data,
                    epochs=epochs,
                    train=name != 'BC'
                )

                for s in score:
                    s.update({"model": name, 'run': run})
                    scores.append(s)

                print("Writing to file", file)
                pd.DataFrame(scores).to_csv(file, index=False)

        return scores

    def plot_learning_curve(self, score_label='Score', loss_label='Loss'):
        file = f"{self.output_dir}/{self.data.dataset}_learning_curve.csv"
        df = pd.read_csv(file).groupby(['model', 'epoch']).describe().reset_index()
        df.columns = [' '.join(col).strip() for col in df.columns]
        for column in ['score', 'loss']:
            df[f"{column} err"] = df[f"{column} std"] / np.sqrt(df["run count"])

        print("--- Score at the last epoch")
        print(df[df['epoch'] == max(df['epoch'])][['model', 'score mean', 'score err']])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.3))
        plt.tight_layout(pad=3, rect=(0, 0, 1, 0.95))
        for key, group in df.groupby('model'):
            args = {'label': key, 'legend': False, 'capsize': 2, 'elinewidth': 1}
            group.plot(x='epoch', y='score mean', yerr='score err', ax=ax1, **args)
            group.plot(x='epoch', y='loss mean', yerr='loss err', ax=ax2, **args)

        ax2.set_yscale('log')
        for ax, label in [(ax1, score_label), (ax2, loss_label)]:
            ax.set_xscale('log')
            ax.set_xlabel('Number of Epochs', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.setp(ax.spines.values(), linewidth=1.5)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(df.model.unique()), prop={"size": 12})

        file = f"{self.output_dir}/{self.data.dataset}_learning_curve.png"
        fig.savefig(file, bbox_inches='tight', format='png', dpi=300)
        print(f"Image saved in {file}")

    def plot_explanation(self, c, batch_size=128, random_state=123):
        torch.manual_seed(random_state)
        X_train, X_test, y_train, y_test = self.data.split(random_state=random_state)
        train_batches, test_data = self.to_torch(X_train, X_test, y_train, y_test, batch_size)

        args = {
            'epochs': 1,
            'train_batches': train_batches,
            'test_data': test_data,
            'eval': False
        }

        net = Quantum(X_train.shape[1], len(np.unique(y_train)), device=self.device)
        w0 = torch.clone(net.born.weight.data)
        w0 = torch.complex(real=w0[0], imag=w0[1]).cpu()

        self.train_and_eval(net=net, **args)
        w1 = torch.clone(net.born.weight.data)
        w1 = torch.complex(real=w1[0], imag=w1[1]).cpu()

        self.train_and_eval(net=net, **args)
        w2 = torch.clone(net.born.weight.data)
        w2 = torch.complex(real=w2[0], imag=w2[1]).cpu()

        top = torch.argsort(w2[:, c].abs(), descending=True)
        names = self.data.features_names

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': 'polar'})
        plt.tight_layout(pad=3, rect=(0, 0, 1, 0.95))
        for linestyle, marker, idx in [('solid', 'o', top[0:10]), ('dotted', 'x', top[-10:])]:
            for i in idx:
                for ax, w in [(ax1, w0), (ax2, w1), (ax3, w2)]:
                    ax.plot([0, w[i, c].angle()], [0, w[i, c].abs()],
                            label=names[i], marker=marker, linestyle=linestyle)

        for ax, title in [(ax1, 'Initial Weights'), (ax2, 'Epoch 1'), (ax3, 'Epoch 2')]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(title, y=0, pad=-25)

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=10, prop={"size": 8.5})

        file = f"{self.output_dir}/{self.data.dataset}_explanation.png"
        fig.savefig(file, bbox_inches='tight', format='png', dpi=300)
        print(f"Image saved in {file}")

    def train_and_eval(self, net, train_batches, test_data, epochs, train=True, eval=True):
        scores = []
        time_train = 0
        n_batches = len(train_batches)
        optimizer = torch.optim.Adam(net.parameters())
        for epoch in range(epochs):
            print(f"doing epoch {epoch + 1}/{epochs}...", end=" " if eval else "\n")
            shuffle = [train_batches[i] for i in torch.randperm(len(train_batches))]
            for batch_idx, (inputs, labels) in enumerate(shuffle):

                if train:
                    net.train()
                    tic = time()

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    self.loss(outputs, labels).backward()
                    optimizer.step()

                    toc = time()
                    time_train += toc - tic

                if eval and (batch_idx == n_batches - 1):
                    net.eval()
                    with torch.no_grad():
                        inputs, labels = test_data
                        outputs = net(inputs)

                        y_true = torch.argmax(labels, dim=1)
                        y_pred = outputs if self.needs_proba else torch.argmax(outputs, dim=1)

                        if self.needs_proba and y_pred.shape[1] == 2:
                            y_pred = y_pred[:, 1]

                        scores.append({
                            'epoch': epoch + (batch_idx + 1) / n_batches,
                            'score': self.score(y_true.cpu(), y_pred.cpu()),
                            'loss': self.loss(outputs, labels).item(),
                            'time_train': time_train,
                        })

                        print("test score:", scores[-1]['score'], "test loss:", scores[-1]['loss'])

        return scores

    def to_torch(self, X_train, X_test, y_train, y_test, batch_size):
        ohe = OneHotEncoder(handle_unknown='ignore')
        y_train = ohe.fit_transform(y_train.reshape(-1, 1)).todense()
        y_test = ohe.transform(y_test.reshape(-1, 1)).todense()
        train_batches = [(X, y) for X, y in self.to_batch(X_train, y_train, batch_size)]
        test_data = (self.to_tensor(X_test), self.to_tensor(y_test))
        return train_batches, test_data

    def to_batch(self, X, y, batch_size):
        idxs = np.arange(X.shape[0])
        for batch_idxs in np.array_split(idxs, batch_size):
            yield self.to_tensor(X[batch_idxs]), self.to_tensor(y[batch_idxs])

    def to_tensor(self, x):
        if sparse.issparse(x):
            x = x.tocoo()
            i = torch.LongTensor(np.vstack((x.row, x.col)))
            v = torch.tensor(x.data, dtype=torch.get_default_dtype())
            return torch.sparse_coo_tensor(i, v, torch.Size(x.shape), device=self.device)
        return torch.tensor(x, dtype=torch.get_default_dtype(), device=self.device)
