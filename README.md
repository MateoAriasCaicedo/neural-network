# Simple neural network

Minimal, pure-Python neural network code for learning purposes. No
NumPy: everything uses plain Python builtins with explicit loops so the
math stays visible.

## Dense layer forward pass

The module [`neural_network.neural_network`](src/neural_network/neural_network.py) implements a
single dense (fully connected) layer. Each neuron in the layer computes
the dot product of its weights with the layer inputs, then adds its
bias:

    ```text
output_i = bias_i + sum_j(weight_ij * input_j)
```

```python
>>> from neural_network.neural_network import layer_neural_network
>>> layer_neural_network([1, 2, 3], [[1, 2, 3], [0, 1, 0]], [0, 1])
[14, 3]
```

- `inputs`: one value per connection into the layer.
- `layer_weight[i]`: weights of neuron `i`; must match the input count.
- `layer_bias[i]`: bias of neuron `i`; same length as `layer_weight`.

A `ValueError` is raised when the shapes are inconsistent (mismatched
bias count, or neuron weights that don't match the number of inputs).

## Running the tests

Tests use [pytest](https://docs.pytest.org) (a dev dependency):

```bash
uv run pytest
```
