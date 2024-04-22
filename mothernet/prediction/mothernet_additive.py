from typing import List

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from mothernet.model_builder import load_model
from mothernet.models.utils import bin_data
from mothernet.models.encoders import get_fourier_features
from mothernet.utils import normalize_data


def extract_additive_model(model, X_train, y_train, device="cpu", inference_device="cpu", pad_zeros=True):
    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    with torch.no_grad():
        n_classes = len(np.unique(y_train))
        n_features = X_train.shape[1]

        ys = torch.Tensor(y_train).to(device)
        xs = torch.Tensor(X_train).to(device)

        if X_train.shape[1] > 100:
            raise ValueError("Cannot run inference on data with more than 100 features")
        if pad_zeros:
            x_all_torch = torch.concat([xs, torch.zeros((X_train.shape[0], 100 - X_train.shape[1]), device=device)], axis=1)
        else:
            x_all_torch = xs
        X_onehot, bin_edges = bin_data(x_all_torch, n_bins=model.n_bins, nan_bin=model.nan_bin)
        if model.input_layer_norm:
            X_onehot = model.input_norm(X_onehot.float())
        if getattr(model, "fourier_features", 0) > 0:
            x_scaled = normalize_data(xs)
            x_fourier = get_fourier_features(x_scaled, model.fourier_features)
            X_onehot = torch.cat([X_onehot, x_fourier], -1)

        x_src = model.encoder(X_onehot.unsqueeze(1).float())
        if getattr(model, 'categorical_embedding', False):
            is_categorical = model._determine_is_categorical(x_src)  # (1, batch_size, num_features)
            x_src += model.is_categorical_encoder(is_categorical)

        if model.y_encoder is None:
            train_x = x_src
        else:
            y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
            if x_src.ndim == 4:
                # baam model, per feature
                y_src = y_src.unsqueeze(-2)
            train_x = x_src + y_src
        assert train_x.shape == x_src.shape
        if hasattr(model, "layers"):
            # baam model
            output = train_x
            for mod in model.layers:
                output = mod(output)
        else:
            output = model.transformer_encoder(train_x)

        if model.marginal_residual in [True, 'True', 'output', 'decoder']:
            class_averages = model.class_average_layer(X_onehot.float().unsqueeze(1), ys.unsqueeze(1))
            # class averages are batch x outputs x features x bins
            # output is batch x features x bins x outputs
            marginals = model.marginal_residual_layer(class_averages)

        if model.marginal_residual == 'decoder':
            weights, biases = model.decoder(output, ys, marginals)
        else:
            weights, biases = model.decoder(output, ys)

        if model.marginal_residual in [True, 'True', 'output', 'decoder']:
            if hasattr(model, "layers") and len(model.layers) == 0:
                weights = marginals.permute(0, 2, 3, 1)
            else:
                weights = weights + marginals.permute(0, 2, 3, 1)
        w = weights.squeeze()[:n_features, :, :n_classes]
        if biases is None:
            b = torch.zeros(n_classes, device=device)
        else:
            b = biases.squeeze()[:n_classes]
        bins_data_space = bin_edges[:n_features]
        # remove extra classes on output layer
        if inference_device == "cpu":
            def detach(x):
                return x.detach().cpu().numpy()
        else:
            def detach(x):
                return x.detach()

    return detach(w), detach(b), detach(bins_data_space)


def predict_with_additive_model(X_train, X_test, weights, biases, bin_edges, nan_bin, inference_device="cpu"):
    additive_components = []
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.shape[1] == len(weights)
    assert weights.shape[0] == X_train.shape[1]
    n_bins = weights.shape[1]
    assert bin_edges.shape == (X_train.shape[1], n_bins - 1)
    if inference_device == "cpu":
        out = np.zeros((X_test.shape[0], weights.shape[-1]))
        for col, bins, w in zip(X_test.T, bin_edges, weights):
            binned = np.searchsorted(bins, col)
            if nan_bin:
                # Put NaN data on the last bin.
                binned[np.isnan(col)] = n_bins - 1
            out += w[binned]
            additive_components.append(w[binned])
        out += biases
        if np.isnan(out).any():
            print("NAN")
            import pdb
            pdb.set_trace()
        from scipy.special import softmax
        return softmax(out / .8, axis=1), additive_components
    elif "cuda" in inference_device:
        raise NotImplementedError('CUDA inference not working')
        mean = torch.Tensor(np.nanmean(X_train, axis=0)).to(inference_device)
        std = torch.Tensor(np.nanstd(X_train, axis=0, ddof=1) + .000001).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        std[torch.isnan(std)] = 1
        X_test_scaled = (torch.Tensor(X_test).to(inference_device) - mean) / std
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        import pdb
        pdb.set_trace()
        raise NotImplementedError
        layers = None
        for i, (b, w) in enumerate(layers):
            out = torch.matmul(out, w) + b
            if i != len(layers) - 1:
                out = torch.relu(out)
        return torch.nn.functional.softmax(out / .8, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown inference_device: {inference_device}")


class MotherNetAdditiveClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cpu", inference_device="cpu", model=None, config=None):
        self.path = path
        self.device = device
        self.inference_device = inference_device
        if model is None and path is None:
            raise ValueError("Either path or model must be provided")
        if model is not None and path is not None:
            raise ValueError("Only one of path or model must be provided")
        if model is not None and config is None:
            raise ValueError("config must be provided if model is provided")
        self.model = model
        self.config = config
        if hasattr(model, "nan_bin"):
            self.nan_bin = model.nan_bin
        else:
            self.nan_bin = False

    def fit(self, X, y, is_categorical: List[bool] = None):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        if self.model is not None:
            model, config = self.model, self.config
        else:
            model, config = load_model(self.path, device=self.device)
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] not in ["additive", "baam"]:
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        try:
            pad_zeros = config['prior']['classification']['pad_zeros']
        except KeyError:
            pad_zeros = True
        w, b, bin_edges = extract_additive_model(model, X, y, device=self.device, inference_device=self.inference_device,
                                                 pad_zeros=pad_zeros)
        self.w_ = w
        self.b_ = b
        self.bin_edges_ = bin_edges
        self.classes_ = le.classes_
        self.pad_zeros = pad_zeros
        return self

    def predict_proba(self, X):
        return predict_with_additive_model(self.X_train_, X, self.w_, self.b_, self.bin_edges_, nan_bin=self.nan_bin,
                                           inference_device=self.inference_device)[0]

    def predict_proba_with_additive_components(self, X):
        return predict_with_additive_model(self.X_train_, X, self.w_, self.b_, self.bin_edges_, nan_bin=self.nan_bin,
                                           inference_device=self.inference_device)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
