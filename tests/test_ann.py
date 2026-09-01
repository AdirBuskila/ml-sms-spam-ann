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
