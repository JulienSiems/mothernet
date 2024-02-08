import random
import time

import numpy as np
import torch
from torch import nn

from tabpfn.utils import (nan_handling_missing_for_a_reason_value, nan_handling_missing_for_no_reason_value,
                          nan_handling_missing_for_unknown_reason_value, normalize_by_used_features_f, normalize_data,
                          remove_outliers, to_ranking_low_mem)
from tabpfn.priors.differentiable_prior import sample_distributions

from .utils import CategoricalActivation, randomize_classes, uniform_int_sampler_f


class BalancedBinarize:
    def __call__(self, x):
        return (x > torch.median(x)).float()


def class_sampler_f(min_, max_):
    def s():
        if random.random() > 0.5:
            return uniform_int_sampler_f(min_, max_)()
        return 2
    return s


class RegressionNormalized:
    def __call__(self, x):
        # x has shape (T,B)

        # TODO: Normalize to -1, 1 or gaussian normal
        maxima = torch.max(x, 0)[0]
        minima = torch.min(x, 0)[0]
        norm = (x - minima) / (maxima-minima)

        return norm


class MulticlassSteps:
    """"Sample piecewise constant functions with random number of steps and random class boundaries"""
    def __init__(self, num_classes, max_steps=10):
        self.num_classes = class_sampler_f(2, num_classes)()
        self.num_steps = np.random.randint(1, max_steps) if max_steps > 1 else 1

    def __call__(self, x):
        # x has shape (T,B,H) ?!
        # x has shape (samples, batch)
        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundary_indices = torch.randint(0, x.shape[0], ((self.num_classes - 1) * (self.num_steps - 1) + 1,), device=x.device)
        class_boundaries_sorted, _ = x[class_boundary_indices].sort(axis=0)
        step_assignments = torch.searchsorted(class_boundaries_sorted.T.contiguous(), x.T.contiguous()).T
        class_assignments = torch.randint(0, self.num_classes, (step_assignments.max() + 1, x.shape[1]), device=x.device)
        classes = torch.gather(class_assignments, 0, step_assignments)
        return classes


class MulticlassRank:
    def __init__(self, num_classes, ordered_p=0.5):
        self.num_classes = class_sampler_f(2, num_classes)()
        self.ordered_p = ordered_p

    def __call__(self, x):
        # x has shape (T,B,H)

        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
        class_boundaries = x[class_boundaries].unsqueeze(1)

        d = (x > class_boundaries).sum(axis=0)

        randomized_classes = torch.rand((d.shape[1], )) > self.ordered_p
        d[:, randomized_classes] = randomize_classes(d[:, randomized_classes], self.num_classes)
        reverse_classes = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
        return d



class ClassificationAdapter:
    # This class samples the number of features actually use (num_features_used), the number of samples
    # adds NaN and potentially categorical features
    # and discretizes the classification output variable
    # It's instantiated anew for each batch that's created
    def __init__(self, base_prior, hyperparameters):

        self.h = sample_distributions(hyperparameters)

        self.base_prior = base_prior
        if self.h['num_classes'] == 0:
            self.class_assigner = RegressionNormalized()
        else:
            if self.h['num_classes'] > 1 and not self.h['balanced']:
                if self.h['multiclass_type'] == 'rank':
                    self.class_assigner = MulticlassRank(
                        self.h['num_classes'], ordered_p=self.h['output_multiclass_ordered_p']
                    )
                elif self.h['multiclass_type'] == 'steps':
                    self.class_assigner =  MulticlassSteps(self.h['num_classes'], self.h['multiclass_max_steps'])
                else:
                    raise ValueError("Unknow Multiclass type")
            elif self.h['num_classes'] == 2 and self.h['balanced']:
                self.class_assigner = BalancedBinarize()
            elif self.h['num_classes'] > 2 and self.h['balanced']:
                raise NotImplementedError("Balanced multiclass training is not possible")

    def drop_for_reason(self, x, v):
        nan_prob_sampler = CategoricalActivation(
            ordered_p=0.0, categorical_p=1.0, keep_activation_size=False,
            num_classes_sampler=lambda: 20)
        d = nan_prob_sampler(x)
        # TODO: Make a different ordering for each activation
        x[d < torch.rand((1,), device=x.device) * 20 * self.h['nan_prob_no_reason'] * random.random()] = v
        return x

    def drop_for_no_reason(self, x, v):
        x[torch.rand(x.shape, device=x.device) < random.random() * self.h['nan_prob_no_reason']] = v
        return x

    def __call__(self, batch_size, n_samples, num_features, device, hyperparameters=None, epoch=None, single_eval_pos=None):
        # num_features is constant for all batches, num_features used is passed down to wrapped priors to change number of features
        args = {'device': device, 'n_samples': n_samples, 'num_features': self.h['num_features_used'], 'batch_size': batch_size, 'epoch': epoch, 'single_eval_pos': single_eval_pos}
        x, y, y_ = self.base_prior.get_batch(hyperparameters=self.h, **args)
        assert x.shape[2] == self.h['num_features_used']

        if self.h['nan_prob_no_reason']+self.h['nan_prob_a_reason']+self.h['nan_prob_unknown_reason'] > 0 and random.random() > 0.5:  # Only one out of two datasets should have nans
            if random.random() < self.h['nan_prob_no_reason']:  # Missing for no reason
                x = self.drop_for_no_reason(x, nan_handling_missing_for_no_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_a_reason'] > 0 and random.random() > 0.5:  # Missing for a reason
                x = self.drop_for_reason(x, nan_handling_missing_for_a_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_unknown_reason'] > 0:  # Missing for unknown reason  and random.random() > 0.5
                if random.random() < self.h['nan_prob_unknown_reason_reason_prior']:
                    x = self.drop_for_no_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))
                else:
                    x = self.drop_for_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))

        # Categorical features
        if 'categorical_feature_p' in self.h and random.random() < self.h['categorical_feature_p']:
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1, 10)), 2)
                m = MulticlassRank(num_unique_features, ordered_p=0.3)
                if random.random() < p:
                    x[:, :, col] = m(x[:, :, col])

        x = remove_outliers(x)
        x, y = normalize_data(x), normalize_data(y)

        # Cast to classification if enabled
        y = self.class_assigner(y).float()

        x = normalize_by_used_features_f(
            x, self.h['num_features_used'], num_features)

        start = time.time()
        # Append empty features if enabled
        x = torch.cat(
            [x, torch.zeros((x.shape[0], x.shape[1], num_features - self.h['num_features_used']),
                            device=device)], -1)


        if torch.isnan(y).sum() > 0:
            print('Nans in target!')

        # if self.h['check_is_compatible']: always true
        for b in range(y.shape[1]):
            is_compatible, N = False, 0
            while not is_compatible and N < 10:
                targets_in_train = torch.unique(y[:single_eval_pos, b], sorted=True)
                targets_in_eval = torch.unique(y[single_eval_pos:, b], sorted=True)

                is_compatible = len(targets_in_train) == len(targets_in_eval) and (
                    targets_in_train == targets_in_eval).all() and len(targets_in_train) > 1

                if not is_compatible:
                    randperm = torch.randperm(x.shape[0])
                    x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
                N = N + 1
            if not is_compatible:
                if not is_compatible:
                    # todo check that it really does this and how many together
                    y[:, b] = -100  # Relies on CE having `ignore_index` set to -100 (default)

        #if self.h['normalize_labels']: this was always true
        for b in range(y.shape[1]):
            valid_labels = y[:, b] != -100
            y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(axis=0).unsqueeze(0).float()

            if y[valid_labels, b].numel() != 0:
                num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                num_classes = num_classes_float.int().item()
                assert num_classes == num_classes_float.item()
                random_shift = torch.randint(0, num_classes, (1,), device=device)
                y[valid_labels, b] = (y[valid_labels, b] + random_shift) % num_classes

        return x, y, y  # x.shape = (T,B,H)

class ClassificationAdapterPrior:
    def __init__(self, base_prior):
        self.base_prior = base_prior

    def get_batch(self, batch_size, n_samples, num_features, device, hyperparameters=None, epoch=None, single_eval_pos=None):
        with torch.no_grad():
            # Sample one n_samples for entire batch
            n_samples = hyperparameters['n_samples_used']() if callable(hyperparameters['n_samples_used']) else n_samples

            args = {'device': device, 'n_samples': n_samples, 'num_features': num_features, 'epoch': epoch, 'single_eval_pos': single_eval_pos}

            x, y, y_ = ClassificationAdapter(self.base_prior, hyperparameters)(batch_size=batch_size, **args)
            x, y, y_ = x.detach(), y.detach(), y_.detach()

        return x, y, y_
