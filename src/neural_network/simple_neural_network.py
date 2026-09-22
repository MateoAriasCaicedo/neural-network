"""A minimal, pure-Python neural network layer — no NumPy.

This module implements the forward pass of a single dense (fully
connected) layer. Each neuron computes the dot product of its weights
with the inputs, then adds its bias:

    output_i = bias_i + sum_j(weight_ij * input_j)

It is written as an educational example: plain Python builtins and
explicit loops keep the math visible, at the cost of performance that a
vectorized library such as NumPy would provide.
"""

from collections.abc import Sequence


def simple_neural_network(
    inputs: Sequence[float],
    layer_weight: Sequence[Sequence[float]],
    layer_bias: Sequence[float],
) -> list[float]:
    """Forward pass through one dense layer.

    For every neuron, computes the weighted sum of the inputs plus the
    neuron's bias:

        output_i = bias_i + sum_j(weight_ij * input_j)

    Args:
        inputs: Signals entering the layer, one per connection.
        layer_weight: One list of weights per neuron in the layer.
            Each inner list must be the same length as ``inputs``.
        layer_bias: One bias value per neuron in the layer. Must be the
            same length as ``layer_weight``.

    Returns:
        A list with one output value per neuron.

    Examples:
        >>> simple_neural_network([1, 2], [[1, 2]], [1])
        [6]

    Raises:
        ValueError: If the biases and weights describe a different
            number of neurons, or if a neuron's weights do not match
            the number of inputs.
    """
    _validate_shapes(inputs, layer_weight, layer_bias)

    output: list[float] = []

    for neuron_weight, neuron_bias in zip(layer_weight, layer_bias):
        neuron_output = neuron_bias

        for weight, input_value in zip(neuron_weight, inputs):
            neuron_output += weight * input_value

        output.append(neuron_output)

    return output


def _validate_shapes(
    inputs: Sequence[float],
    layer_weight: Sequence[Sequence[float]],
    layer_bias: Sequence[float],
) -> None:
    """Raise ValueError if the layer's shapes are inconsistent."""
    if len(layer_weight) != len(layer_bias):
        raise ValueError(
            "layer_weight and layer_bias must describe the same number of "
            f"neurons, got {len(layer_weight)} weight rows and "
            f"{len(layer_bias)} biases"
        )

    for index, neuron_weight in enumerate(layer_weight):
        if len(neuron_weight) != len(inputs):
            raise ValueError(
                f"neuron {index} expects {len(neuron_weight)} inputs, "
                f"but {len(inputs)} were provided"
            )
