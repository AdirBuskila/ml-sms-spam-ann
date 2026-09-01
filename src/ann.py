"""A feed-forward neural network (multi-layer perceptron) written from scratch in NumPy.

Architecture  x -> [hidden layer(s) with relu/tanh/sigmoid] -> 1 sigmoid unit = P(spam)
Loss          binary cross-entropy (optionally class-weighted) + 0.5 * l2 * sum(W^2)
Training      mini-batch stochastic gradient descent; gradients come from back-propagation.
hidden_layers=() removes every hidden layer, which makes this exactly logistic regression.
"""
from __future__ import annotations

import numpy as np

_ACTIVATIONS = ("relu", "tanh", "sigmoid")
_EPS = 1e-12
_MAX_ABS_WEIGHT = 1e6      # weights beyond this magnitude only ever mean SGD has blown up


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # 1 / (1 + e^-z) computed as exp(-log(1 + e^-z)); logaddexp never overflows.
    return np.exp(-np.logaddexp(0.0, -z))


class NeuralNetwork:
    """Binary classifier with scikit-learn-like fit / predict / predict_proba.

    Hyperparameters
    ---------------
    hidden_layers : tuple of ints, units per hidden layer; () = logistic regression
    activation    : "relu" | "tanh" | "sigmoid" for the hidden layers (output is always sigmoid)
    learning_rate : SGD step size
    epochs        : passes over the training set
    batch_size    : rows per SGD step
    l2            : L2 penalty on the weights (not on biases)
    class_weight  : None or "balanced" (positive rows weighted by n_neg / n_pos in the loss)
    seed          : controls weight initialisation and mini-batch shuffling
    """

    def __init__(self, hidden_layers=(32,), activation="relu", learning_rate=0.1, epochs=30,
                 batch_size=32, l2=0.0, class_weight=None, seed=0, verbose=False):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        if class_weight not in (None, "balanced"):
            raise ValueError(f"class_weight must be None or 'balanced', got {class_weight!r}")
        if learning_rate <= 0 or epochs < 1 or batch_size < 1 or l2 < 0:
            raise ValueError("learning_rate > 0, epochs >= 1, batch_size >= 1, l2 >= 0 required")
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.seed = seed
        self.verbose = verbose

    def __repr__(self) -> str:
        return (f"NeuralNetwork(hidden_layers={self.hidden_layers}, activation={self.activation!r}, "
                f"learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size}, "
                f"l2={self.l2}, class_weight={self.class_weight!r}, seed={self.seed})")

    # ------------------------------------------------------------------ parameters
    def _init_weights(self, n_features: int) -> np.random.Generator:
        """He initialisation for relu, Glorot/Xavier otherwise; zero biases. Returns the RNG."""
        rng = np.random.default_rng(self.seed)
        sizes = [n_features, *self.hidden_layers, 1]
        self.weights_, self.biases_ = [], []
        for layer, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            is_output = layer == len(sizes) - 2
            if self.activation == "relu" and not is_output:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights_.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))
        return rng

    # ------------------------------------------------------------------ forward pass
    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "tanh":
            return np.tanh(z)
        return _sigmoid(z)

    def _activate_grad(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Derivative of the hidden activation, from the pre-activation z or the activation a."""
        if self.activation == "relu":
            return (z > 0.0).astype(z.dtype)
        if self.activation == "tanh":
            return 1.0 - a * a
        return a * (1.0 - a)

    def _forward(self, X: np.ndarray):
        """Return (activations, pre_activations); activations[0] is X, activations[-1] is P(spam)."""
        activations, pre_activations = [X], []
        a, last = X, len(self.weights_) - 1
        for layer, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b
            a = _sigmoid(z) if layer == last else self._activate(z)
            pre_activations.append(z)
            activations.append(a)
        return activations, pre_activations

    # ------------------------------------------------------------------ loss and gradients
    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        w = np.ones(len(y), dtype=np.float64)
        if self.class_weight == "balanced":
            n_pos = float(np.sum(y == 1))
            n_neg = float(len(y) - n_pos)
            if n_pos > 0:
                w[y == 1] = n_neg / n_pos
        return w

    def _loss(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> float:
        p = self._forward(X)[0][-1].ravel()
        p = np.clip(p, _EPS, 1.0 - _EPS)
        bce = -np.sum(sample_weight * (y * np.log(p) + (1.0 - y) * np.log1p(-p))) / len(y)
        reg = 0.5 * self.l2 * sum(float(np.sum(W * W)) for W in self.weights_)
        return float(bce + reg)

    def _backward(self, activations, pre_activations, y: np.ndarray, sample_weight: np.ndarray):
        """Back-propagation. Returns gradients of _loss w.r.t. every weight matrix and bias vector."""
        n = len(y)
        n_layers = len(self.weights_)
        p = activations[-1].ravel()
        # dLoss/dz for sigmoid + cross-entropy collapses to (p - y); weights and 1/n come from the mean.
        delta = ((p - y) * sample_weight / n)[:, None]
        grads_W, grads_b = [None] * n_layers, [None] * n_layers
        for layer in range(n_layers - 1, -1, -1):
            grads_W[layer] = activations[layer].T @ delta + self.l2 * self.weights_[layer]
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = (delta @ self.weights_[layer].T) * self._activate_grad(pre_activations[layer - 1], activations[layer])
        return grads_W, grads_b

    # ------------------------------------------------------------------ public API
    def fit(self, X, y) -> "NeuralNetwork":
        """Mini-batch SGD on the (weighted) cross-entropy. Records the full-data loss per epoch."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must contain only 0 and 1")
        y = y.astype(np.float64)
        rng = self._init_weights(X.shape[1])
        sample_weight = self._sample_weights(y)
        self.loss_history_ = []
        self.diverged_ = False
        n = len(y)
        # A too-large learning rate makes the weights overflow; NumPy would then warn on every
        # operation. We silence those warnings here and instead detect the blow-up once per epoch.
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(self.epochs):
                order = rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    idx = order[start:start + self.batch_size]
                    activations, pre_activations = self._forward(X[idx])
                    grads_W, grads_b = self._backward(activations, pre_activations, y[idx], sample_weight[idx])
                    for layer in range(len(self.weights_)):
                        self.weights_[layer] -= self.learning_rate * grads_W[layer]
                        self.biases_[layer] -= self.learning_rate * grads_b[layer]
                loss = self._loss(X, y, sample_weight)
                self.loss_history_.append(loss)
                if self.verbose:
                    print(f"epoch {epoch + 1:3d}/{self.epochs}  loss = {loss:.4f}")
                exploded = any(not np.isfinite(W).all() or np.abs(W).max() > _MAX_ABS_WEIGHT for W in self.weights_)
                if not np.isfinite(loss) or exploded:
                    self.diverged_ = True          # training blew up: stop, keep what we have
                    if self.verbose:
                        print("training diverged (non-finite loss or exploding weights) - stopping early")
                    break
        self.n_features_in_ = X.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("call fit(X, y) before predicting")

    def predict_proba(self, X) -> np.ndarray:
        """P(spam) for every row, shape (n,)."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected shape (n, {self.n_features_in_}), got {X.shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            p = self._forward(X)[0][-1].ravel()
        # a diverged model has NaN weights; report "no evidence of spam" instead of NaN
        return np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """1 (spam) where P(spam) >= threshold, else 0."""
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def n_parameters_(self) -> int:
        return int(sum(W.size for W in self.weights_) + sum(b.size for b in self.biases_))


def gradient_check(model: NeuralNetwork, X, y, eps: float = 1e-5) -> float:
    """Compare back-propagation with central finite differences on every single parameter.

    For each weight matrix / bias vector the numerical gradient G_num is assembled element by
    element from (loss(theta + eps) - loss(theta - eps)) / (2 eps) and compared with the analytic
    gradient G as  ||G_num - G|| / max(||G_num|| + ||G||, 1e-8).  The function returns the worst
    of these relative errors. Around 1e-8 means backprop is correct; 1e-2 means a bug.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if not hasattr(model, "weights_"):
        model._init_weights(X.shape[1])
    w = model._sample_weights(y)
    activations, pre_activations = model._forward(X)
    grads_W, grads_b = model._backward(activations, pre_activations, y, w)
    worst = 0.0
    for params, grads in ((model.weights_, grads_W), (model.biases_, grads_b)):
        for P, G in zip(params, grads):
            numeric = np.zeros_like(P)
            for idx in np.ndindex(P.shape):
                original = P[idx]
                P[idx] = original + eps
                loss_plus = model._loss(X, y, w)
                P[idx] = original - eps
                loss_minus = model._loss(X, y, w)
                P[idx] = original
                numeric[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            rel = np.linalg.norm(numeric - G) / max(np.linalg.norm(numeric) + np.linalg.norm(G), 1e-8)
            worst = max(worst, float(rel))
    return worst
