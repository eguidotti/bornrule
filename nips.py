from nips.experiment import Experiment
from nips.metrics import log_loss
from sklearn.metrics import accuracy_score

# Setup experiment:
# - dataset: one of '20ng', 'r8', 'r52'
# - loss: the loss used to train neural networks
# - score: the metric to evaluate the classification results
# - needs_proba: True if score requires probabilities rather than predicted classes
# - data_dir: the directory where data are to be downloaded
# - output_dir: the directory where results are to be saved
ex = Experiment(dataset='20ng', loss=log_loss, score=accuracy_score, needs_proba=False)

# Summary statistics
ex.data.summary()

# Timing CPU and GPU
ex.timing_gpu(runs=5)
ex.timing_cpu(runs=5)
ex.plot_timing(score_label='Accuracy Score')

# Table cross-validation times and scores
ex.cross_validation(runs=1)

# Run and plot the ablation study
ex.ablation_study(runs=1)
ex.plot_ablation()

# Table top 10 features for Born classifier
ex.table_explanation(top=10)

# Train the networks and plot the learning curves
scores = ex.learning_curve(epochs=100, runs=5, batch_size=128)
ex.plot_learning_curve(score_label="Accuracy Score", loss_label="Log-Loss")

# Plot explanation of Born layer for class 9 (baseball)
ex.plot_explanation(c=9, batch_size=128, random_state=0)

# R8 dataset
ex = Experiment(dataset='r8', loss=log_loss, score=accuracy_score, needs_proba=False)
ex.timing_gpu(runs=5)

# R52 dataset
ex = Experiment(dataset='r52', loss=log_loss, score=accuracy_score, needs_proba=False)
ex.timing_gpu(runs=5)
