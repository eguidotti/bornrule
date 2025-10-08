import torch
from nips.metrics import l1_loss
from nips.experiment import Experiment
from sklearn.metrics import accuracy_score

# Setup experiments:
# - loss: the loss used to train neural networks
# - score: the metric to evaluate the classification results
# - needs_proba: True if score requires probabilities rather than predicted classes
# - device: the torch device (CPU/GPU)
# - data_dir: the directory where data are to be downloaded
# - output_dir: the directory where results are to be saved
kwargs = {
    'loss': l1_loss,
    'score': accuracy_score,
    'needs_proba': False,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'data_dir': 'data',
    'output_dir': 'results',
}


# R8 dataset
r8 = Experiment(dataset='r8', **kwargs)
r8.data.summary()

# Timing
r8.timing_gpu(runs=10)
r8.timing_cpu(runs=10)
r8.plot_timing(score_label='Accuracy Score')

# Train the networks and plot the learning curves
r8.learning_curve(epochs=100, runs=10, batch_size=128)
r8.plot_learning_curve(score_label="Accuracy Score", loss_label="L1-Loss")


# R52 dataset
r52 = Experiment(dataset='r52', **kwargs)
r52.data.summary()

# Timing
r52.timing_gpu(runs=10)
r52.timing_cpu(runs=10)
r52.plot_timing(score_label='Accuracy Score')

# Train the networks and plot the learning curves
r52.learning_curve(epochs=100, runs=10, batch_size=128)
r52.plot_learning_curve(score_label="Accuracy Score", loss_label="L1-Loss")


# 20NG dataset
ng = Experiment(dataset='20ng', **kwargs)
ng.data.summary()

# Timing GPU
ng.timing_gpu(runs=10)
ng.timing_cpu(runs=10)
ng.plot_timing(score_label='Accuracy Score')

# Train the networks and plot the learning curves
ng.learning_curve(epochs=100, runs=10, batch_size=128)
ng.plot_learning_curve(score_label="Accuracy Score", loss_label="L1-Loss")

# Plot explanation of Born layer for class 9 (baseball)
ng.plot_explanation(c=9, batch_size=128, random_state=51)

# Table top 10 features for Born classifier
ng.table_explanation(top=10)

# Ablation study
ng.ablation_study()
ng.plot_ablation()

# Table cross-validation times and scores (takes 24-48 hours)
ng.cross_validation()
