from pytest import raises
from neural_network.neural_network import layer_neural_network


class TestCorrectness:
    def test_single_neuron(self):
        assert layer_neural_network([1, 2], [[1, 2]], [1]) == [6]

    def test_multiple_neurons(self):
        result = layer_neural_network(
            [1, 2, 3], [[1, 2, 3], [0, 1, 0], [1, 0, 0]], [0, 1, -1]
        )
        assert result == [14, 3, 0]

    def test_negative_values(self):
        assert layer_neural_network([-1, 2.5], [[-0.5, 0.25]], [1.5]) == [2.625]

    def test_zero_inputs(self):
        assert layer_neural_network([0, 0], [[1, 2]], [5]) == [5]

    def test_zero_weights_and_bias(self):
        assert layer_neural_network([3, -4], [[0, 0]], [0]) == [0]

    def test_fractional_inputs(self):
        assert layer_neural_network([0.5, 0.25], [[2, 4]], [0]) == [2.0]

    def test_empty_inputs_matches_bias(self):
        assert layer_neural_network([], [[]], [7]) == [7]

    def test_empty_layer_returns_empty(self):
        assert layer_neural_network([1, 2], [], []) == []

    def test_does_not_mutate_arguments(self):
        inputs = [1, 2]
        weights = [[1, 2]]
        biases = [1]
        layer_neural_network(inputs, weights, biases)
        assert inputs == [1, 2]
        assert weights == [[1, 2]]
        assert biases == [1]


class TestValidation:
    def test_too_few_biases_raises(self):
        with raises(ValueError, match="same number of neurons"):
            layer_neural_network([1, 2], [[1, 2], [3, 4]], [0])

    def test_too_many_biases_raises(self):
        with raises(ValueError, match="same number of neurons"):
            layer_neural_network([1, 2], [[1, 2]], [0, 1])

    def test_weight_count_mismatch_raises(self):
        with raises(ValueError, match="expects 2 inputs"):
            layer_neural_network([1, 2, 3], [[1, 2]], [0])
