# neural_network

A pure-Python neural networks project for learning and experimentation.

Everything is built with plain Python builtins and explicit loops — no
NumPy — so the underlying math stays visible and understandable. This
makes the code ideal for studying how neural networks work before moving
on to vectorized libraries.

## Features

- Minimal, readable implementations with no external dependencies
- Type-annotated public APIs
- Shape validation with clear `ValueError` messages
- Tested with [pytest](https://docs.pytest.org)

## Project structure

```text
src/neural_network/
    simple_neural_network.py   # forward pass of a simple dense layer
tests/
    test_simple_neural_network.py
```

## Current components

The [`simple_neural_network`](src/neural_network/simple_neural_network.py)
module currently provides a forward pass for a single dense (fully
connected) layer, the building block of any neural network.

Over time it will grow into a fuller toolkit, adding activation
functions, multi-layer models, backpropagation, and training
utilities.

## Getting started

Install the project and its dependencies:

```bash
uv sync
```

Basic usage:

```python
from neural_network.simple_neural_network import simple_neural_network

output = simple_neural_network([1, 2, 3], [[1, 2, 3], [0, 1, 0]], [0, 1])
print(output)  # [14, 3]
```

## Running the tests

```bash
uv run pytest
```