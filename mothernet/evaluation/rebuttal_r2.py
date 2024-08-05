import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from mothernet.datasets import linear_correlated_logistic_regression
from mothernet.evaluation.imbalanced_data import eval_gamformer_and_ebm

plt.style.use(['science', 'no-latex', 'light'])
plt.rcParams["figure.constrained_layout.use"] = True
plt.savefig('mimic_2_shape_functions.pdf', dpi=300, bbox_inches='tight')


def true_main_effect(X, weight):
    return weight * X


def estimate_kendall_tau(true_func, estimated_func, X, weight):
    true_effect = true_func(X, weight)
    estimated_effect = estimated_func(X)
    return spearmanr(true_effect, estimated_effect).statistic


def interpolate_shape_function(bin_edges, weights, model_name, X_train, feature_name):
    bin_edges = np.array(bin_edges).flatten()
    weights = np.array(weights)

    if model_name == 'EBM':
        if len(bin_edges) == len(weights) + 1:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            raise ValueError(f"Unexpected shapes for EBM: bin_edges ({len(bin_edges)}), weights ({len(weights)})")
    elif model_name == 'GAMformer':
        if weights.shape[1] > 1:
            weights = weights[:, 1]
        else:
            weights = weights.flatten()
        weights = weights[0:-1] - weights[0:-1].mean(axis=-1)
        bin_centers = bin_edges
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Normalize weights
    weights = weights - weights.mean()

    def shape_function(X):
        return np.interp(X, bin_centers, weights)

    return shape_function


n_features = 3
sample_sizes = np.linspace(32, 150, 15, dtype=int)
n_splits = 5
true_weights = np.linspace(-1, 1, n_features)

results = {model: {f'x_{i + 1}': [] for i in range(n_features)} for model in ['EBM', 'GAMformer']}

for n_datapoints in sample_sizes:
    X, y = linear_correlated_logistic_regression(n_features=n_features, n_tasks=1, n_datapoints=n_datapoints,
                                                 sampling_correlation=0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    column_names = [f'x_{i + 1}' for i in range(n_features)]
    model_results = eval_gamformer_and_ebm('logistic regression', X_train, y_train, X_test, y_test, n_splits=n_splits,
                                           column_names=column_names, record_shape_functions=True)

    for model_idx, model_name in enumerate(['EBM', 'GAMformer']):
        for feature_idx in range(n_features):
            feature_name = f'x_{feature_idx + 1}'
            kendall_tau_scores = []

            for ensemble_idx in range(n_splits):
                bin_edges = model_results[model_idx]['bin_edges'][ensemble_idx][feature_name]
                weights = model_results[model_idx]['w'][ensemble_idx][feature_name]

                try:
                    estimated_func = interpolate_shape_function(bin_edges, weights, model_name,
                                                                pd.DataFrame(X_train, columns=column_names),
                                                                feature_name)

                    X_feature = X_test[:, feature_idx]
                    true_weight = true_weights[feature_idx]

                    kendall_tau = estimate_kendall_tau(true_main_effect, estimated_func, X_feature, true_weight)
                    kendall_tau_scores.append(kendall_tau)
                except ValueError as e:
                    print(f"Error for {model_name}, feature {feature_name}, ensemble {ensemble_idx}: {str(e)}")
                    print(f"bin_edges shape: {np.array(bin_edges).shape}, weights shape: {np.array(weights).shape}")

            if kendall_tau_scores:
                results[model_name][feature_name].append(np.mean(kendall_tau_scores))
            else:
                results[model_name][feature_name].append(np.nan)
fig, axs = plt.subplots(1, 2, figsize=(4, 2), sharey=True)  # Increased width to accommodate legend
for i, feature in enumerate([f'x_{i + 1}' for i in [0, 2]]):
    ax = axs[i]
    for model in ['EBM', 'GAMformer']:
        ax.plot(sample_sizes, results[model][feature], label=model)
    ax.set_xlabel('Number of Samples', fontsize=12)
    if i == 0:
        ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title(rf'${feature}$', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    if i == 0 or i == 1:
        ax.set_ylim(0.89, 1.025)

# Move legend outside the plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title='Model', title_fontsize=10, fontsize=10,
           loc='center right', bbox_to_anchor=(1.23, 0.5))

plt.tight_layout()
# Adjust the layout to make room for the legend
plt.subplots_adjust(right=0.9)

plt.savefig('spearman_vs_sample_size_by_feature.pdf', dpi=300, bbox_inches='tight')
plt.show()


