import numpy


def single_layer_nn(inputs, weight, bias):
    return numpy.dot(numpy.array(inputs), numpy.array(weight).T) + bias


def multi_layer_nn(inputs, weights, biases):
    current_input = inputs

    for weight, bias in zip(weights, biases):
        current_input = single_layer_nn(current_input, weight, bias)

    return current_input
