import numpy as np
import pytest

from src.ann import NeuralNetwork, _sigmoid, gradient_check


def toy_problem(n=12, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.4).astype(np.float64)
    return X, y


class TestConstruction:
    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            NeuralNetwork(activation="swish")
        with pytest.raises(ValueError):
            NeuralNetwork(class_weight="heavy")
        with pytest.raises(ValueError):
            NeuralNetwork(learning_rate=0)
        with pytest.raises(ValueError):
            NeuralNetwork(l2=-1.0)

    def test_weight_shapes_follow_the_layer_sizes(self):
        net = NeuralNetwork(hidden_layers=(5, 3))
        net._init_weights(n_features=7)
        assert [W.shape for W in net.weights_] == [(7, 5), (5, 3), (3, 1)]
        assert [b.shape for b in net.biases_] == [(5,), (3,), (1,)]
        assert all((b == 0).all() for b in net.biases_)

    def test_no_hidden_layer_is_a_single_weight_vector(self):
        net = NeuralNetwork(hidden_layers=())
        net._init_weights(n_features=4)
        assert [W.shape for W in net.weights_] == [(4, 1)]

    def test_repr_lists_hyperparameters(self):
        text = repr(NeuralNetwork(hidden_layers=(8,), activation="tanh", learning_rate=0.3))
        assert "hidden_layers=(8,)" in text and "activation='tanh'" in text and "learning_rate=0.3" in text


class TestForward:
    def test_output_is_a_probability_per_row(self):
        X, _ = toy_problem()
        net = NeuralNetwork(hidden_layers=(4,))
        net._init_weights(X.shape[1])
        activations, pre_activations = net._forward(X)
        assert len(activations) == 3 and len(pre_activations) == 2
        out = activations[-1]
        assert out.shape == (len(X), 1)
        assert ((out > 0) & (out < 1)).all()

    def test_sigmoid_is_stable_for_huge_inputs(self):
        np.testing.assert_allclose(_sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0.0, 0.5, 1.0], atol=1e-12)


class TestSampleWeights:
    def test_balanced_up_weights_the_positive_class(self):
        y = np.array([1, 0, 0, 0], dtype=float)
        assert NeuralNetwork(class_weight=None)._sample_weights(y).tolist() == [1, 1, 1, 1]
        assert NeuralNetwork(class_weight="balanced")._sample_weights(y).tolist() == [3, 1, 1, 1]


class TestGradients:
    def test_logistic_regression_gradient_has_the_textbook_form(self):
        X, y = toy_problem()
        net = NeuralNetwork(hidden_layers=())
        net._init_weights(X.shape[1])
        activations, pre = net._forward(X)
        (gW,), (gb,) = net._backward(activations, pre, y, np.ones_like(y))
        p = activations[-1].ravel()
        np.testing.assert_allclose(gW.ravel(), X.T @ (p - y) / len(y), atol=1e-12)
        np.testing.assert_allclose(gb, [(p - y).mean()], atol=1e-12)

    @pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
    @pytest.mark.parametrize("hidden", [(), (5,), (6, 4)])
    @pytest.mark.parametrize("l2", [0.0, 0.01])
    @pytest.mark.parametrize("class_weight", [None, "balanced"])
    def test_backprop_matches_numerical_gradient(self, activation, hidden, l2, class_weight):
        X, y = toy_problem(seed=1)
        net = NeuralNetwork(hidden_layers=hidden, activation=activation, l2=l2, class_weight=class_weight, seed=3)
        assert gradient_check(net, X, y) < 1e-6

    def test_gradient_check_detects_a_wrong_gradient(self):
        """The check must be able to fail: sabotage one gradient and make sure it notices."""
        X, y = toy_problem(seed=2)
        net = NeuralNetwork(hidden_layers=(4,), seed=0)
        original_backward = net._backward

        def wrong_backward(*args, **kwargs):
            grads_W, grads_b = original_backward(*args, **kwargs)
            grads_W[0] = grads_W[0] * 1.5
            return grads_W, grads_b

        net._backward = wrong_backward
        assert gradient_check(net, X, y) > 1e-2


def blobs(n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2.0, 1.0, size=(n_per_class, 2)), rng.normal(2.0, 1.0, size=(n_per_class, 2))])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def xor_problem(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    return X, y


class TestTraining:
    def test_fit_returns_self_and_records_decreasing_loss(self):
        X, y = blobs()
        net = NeuralNetwork(hidden_layers=(), learning_rate=0.5, epochs=50, seed=0)
        assert net.fit(X, y) is net
        assert len(net.loss_history_) == 50
        assert net.loss_history_[-1] < net.loss_history_[0]

    def test_separable_blobs_are_learned_perfectly(self):
        X, y = blobs()
        net = NeuralNetwork(hidden_layers=(), learning_rate=0.5, epochs=100, seed=0).fit(X, y)
        assert (net.predict(X) == y).mean() == 1.0

    def test_hidden_layer_solves_xor_but_logistic_regression_cannot(self):
        X, y = xor_problem()
        linear = NeuralNetwork(hidden_layers=(), learning_rate=0.5, epochs=200, seed=0).fit(X, y)
        mlp = NeuralNetwork(hidden_layers=(16,), activation="tanh", learning_rate=0.5, epochs=300,
                            batch_size=16, seed=0).fit(X, y)
        assert (linear.predict(X) == y).mean() < 0.75
        assert (mlp.predict(X) == y).mean() >= 0.95

    def test_predict_proba_shape_range_and_threshold(self):
        X, y = blobs()
        net = NeuralNetwork(hidden_layers=(4,), epochs=5).fit(X, y)
        p = net.predict_proba(X)
        assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()
        assert set(np.unique(net.predict(X))) <= {0, 1}
        assert net.predict(X, threshold=1.01).sum() == 0

    def test_accepts_float32_features_and_int_labels(self):
        X, y = blobs()
        net = NeuralNetwork(hidden_layers=(4,), epochs=3).fit(X.astype(np.float32), y.astype(np.int64))
        assert net.predict(X.astype(np.float32)).shape == (len(X),)

    def test_same_seed_same_model_different_seed_different_model(self):
        X, y = blobs()
        a = NeuralNetwork(hidden_layers=(4,), epochs=3, seed=1).fit(X, y)
        b = NeuralNetwork(hidden_layers=(4,), epochs=3, seed=1).fit(X, y)
        c = NeuralNetwork(hidden_layers=(4,), epochs=3, seed=2).fit(X, y)
        for Wa, Wb in zip(a.weights_, b.weights_):
            np.testing.assert_array_equal(Wa, Wb)
        assert any(not np.array_equal(Wa, Wc) for Wa, Wc in zip(a.weights_, c.weights_))

    def test_balanced_class_weight_raises_recall_on_imbalanced_data(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0.0, 1.0, size=(190, 2)), rng.normal(1.5, 1.0, size=(10, 2))])
        y = np.array([0] * 190 + [1] * 10)
        plain = NeuralNetwork(hidden_layers=(), learning_rate=0.3, epochs=30, seed=0).fit(X, y)
        balanced = NeuralNetwork(hidden_layers=(), learning_rate=0.3, epochs=30, seed=0,
                                 class_weight="balanced").fit(X, y)
        assert balanced.predict(X)[y == 1].mean() > plain.predict(X)[y == 1].mean()

    def test_input_validation(self):
        X, y = blobs()
        with pytest.raises(ValueError):
            NeuralNetwork().fit(X, np.where(y == 1, 2, 0))       # labels must be 0/1
        with pytest.raises(ValueError):
            NeuralNetwork().fit(X[:10], y)                        # length mismatch
        with pytest.raises(ValueError):
            NeuralNetwork().fit(X.ravel(), y)                     # X must be 2-D
        with pytest.raises(RuntimeError):
            NeuralNetwork().predict(X)                            # not fitted
        net = NeuralNetwork(epochs=1).fit(X, y)
        with pytest.raises(ValueError):
            net.predict(X[:, :1])                                 # wrong number of features

    def test_n_parameters(self):
        net = NeuralNetwork(hidden_layers=(5, 3)).fit(*blobs())
        assert net.n_parameters_ == (2 * 5 + 5) + (5 * 3 + 3) + (3 * 1 + 1)

    def test_divergence_is_detected_silently_and_predictions_stay_finite(self):
        """An absurd learning rate makes SGD blow up: fit must stop, flag it, and not emit NumPy warnings."""
        import warnings

        X, y = blobs()
        X = X * 1000.0                                            # huge inputs + huge steps => overflow
        net = NeuralNetwork(hidden_layers=(32, 32), learning_rate=1e6, epochs=50, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)        # any RuntimeWarning fails the test
            net.fit(X, y)
            p = net.predict_proba(X)
        assert net.diverged_ is True
        assert len(net.loss_history_) < 50                        # stopped early
        assert np.isfinite(p).all() and set(np.unique(net.predict(X))) <= {0, 1}

    def test_healthy_training_is_not_flagged_as_diverged(self):
        net = NeuralNetwork(hidden_layers=(4,), epochs=5).fit(*blobs())
        assert net.diverged_ is False
