import numpy


def simple_neural_network(
    inputs,
    weights,
    biases,
):
    return numpy.dot(inputs, numpy.array(weights).T) + biases
