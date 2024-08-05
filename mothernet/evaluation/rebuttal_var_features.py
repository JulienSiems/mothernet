import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pickle

from mothernet.datasets import linear_correlated_logistic_regression
from mothernet.evaluation.imbalanced_data import eval_gamformer_and_ebm

plt.style.use(['science', 'no-latex', 'light'])
plt.rcParams["figure.constrained_layout.use"] = True


def generate_data(n_samples, n_features, seed):
    X, y = linear_correlated_logistic_regression(
        n_features=n_features, n_tasks=1, n_datapoints=n_samples, sampling_correlation=0.0, random_state=seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def evaluate_models(X_train, y_train, X_test, y_test, n_features):
    column_names = [f'x_{i}' for i in range(n_features)]
    results = eval_gamformer_and_ebm('toy_example', X_train, y_train, X_test, y_test,
                                     n_splits=5, column_names=column_names)
    return np.mean(results[0]['test_node_gam_scores']), np.mean(results[1]['test_node_gam_scores'])

def process_combination(n_samples, n_features, n_seeds):
    ebm_scores = []
    gamformer_scores = []
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = generate_data(n_samples, n_features, seed)
        ebm_score, gamformer_score = evaluate_models(X_train, y_train, X_test, y_test, n_features)
        ebm_scores.append(ebm_score)
        gamformer_scores.append(gamformer_score)
    return np.mean(ebm_scores), np.mean(gamformer_scores)


# Define ranges for samples and features
sample_sizes = [32, 64, 128, 256, 512, 1024]
feature_counts = [2, 4, 8, 16, 32, 64]
n_seeds = 3

# Parallel processing of combinations
results = Parallel(n_jobs=-1)(delayed(process_combination)(n_samples, n_features, n_seeds)
                              for n_samples in sample_sizes
                              for n_features in feature_counts)

# Reshape results
ebm_results = np.array([r[0] for r in results]).reshape(len(sample_sizes), len(feature_counts))
gamformer_results = np.array([r[1] for r in results]).reshape(len(sample_sizes), len(feature_counts))

# Save results
with open('model_comparison_results.pkl', 'wb') as f:
    pickle.dump({
        'ebm_results': ebm_results,
        'gamformer_results': gamformer_results,
        'sample_sizes': sample_sizes,
        'feature_counts': feature_counts,
        'n_seeds': n_seeds
    }, f)

# Plotting function (separated for reusability)
def plot_results(ebm_results, gamformer_results, sample_sizes, feature_counts, n_seeds):
    # Create a custom colormap
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.4))

    # Determine the global min and max for consistent color scaling
    vmin = min(ebm_results.min(), gamformer_results.min())
    vmax = max(ebm_results.max(), gamformer_results.max())

    # Function to create heatmap
    def create_heatmap(ax, data, title, idx):
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        # Remove ticks but keep labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Set x-axis labels
        ax.set_xticks(np.arange(len(feature_counts)), labels=feature_counts)
        ax.set_xlabel('Number of Features', fontweight='bold')

        if idx == 0:
            ax.set_ylabel('Number of Samples', fontweight='bold')
            # Set y-axis labels for the first plot only
            ax.set_yticks(np.arange(len(sample_sizes)), labels=sample_sizes)
        ax.set_title(title, fontweight='bold')

        # Add text annotations
        for i in range(len(sample_sizes)):
            for j in range(len(feature_counts)):
                text = ax.text(j, i, f"{data[i, j]:.2f}",
                               ha="center", va="center", color="black", fontweight='bold')

        return im

    im1 = create_heatmap(ax1, ebm_results, "EBM", idx=0)
    im2 = create_heatmap(ax2, gamformer_results, "GAMformer", idx=1)

    # Add a colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', pad=0.02)
    cbar.set_label('AUC-ROC Score', rotation=270, labelpad=20, fontweight='bold')

    plt.savefig('model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

import pickle

# Load the saved results
with open('model_comparison_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Plot the results
plot_results(results['ebm_results'], results['gamformer_results'],
             results['sample_sizes'], results['feature_counts'], results['n_seeds'])


