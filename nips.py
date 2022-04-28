from nips.experiment import Experiment
from nips.metrics import log_loss
from sklearn.metrics import accuracy_score

# Setup experiments:
# - dataset: one of '20ng', 'r8', 'r52'
# - loss: the loss used to train neural networks
# - score: the metric to evaluate the classification results
# - needs_proba: True if score requires probabilities rather than predicted classes
# - data_dir: the directory where data are to be downloaded
# - output_dir: the directory where results are to be saved
ng = Experiment(dataset='20ng', loss=log_loss, score=accuracy_score, needs_proba=False)
ng.data.summary()

# R8 dataset
r8 = Experiment(dataset='r8', loss=log_loss, score=accuracy_score, needs_proba=False)
r8.data.summary()

# R52 dataset
r52 = Experiment(dataset='r52', loss=log_loss, score=accuracy_score, needs_proba=False)
r52.data.summary()

# Timing GPU
ng.timing_gpu(runs=5)
r8.timing_gpu(runs=5)
r52.timing_gpu(runs=5)

# Timing CPU
ng.timing_cpu(runs=5)
ng.plot_timing(score_label='Accuracy Score')

# Table top 10 features for Born classifier
ng.table_explanation(top=10)

# Plot explanation of Born layer for class 9 (baseball)
ng.plot_explanation(c=9, batch_size=128, random_state=0)

# Train the networks and plot the learning curves
scores = ng.learning_curve(epochs=1000, runs=5, batch_size=128)
ng.plot_learning_curve(score_label="Accuracy Score", loss_label="Log-Loss")

# Run and plot the ablation study
ng.ablation_study()
ng.plot_ablation()

# Table cross-validation times and scores (takes 24-48 h)
ng.cross_validation()
