#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
# Author: Jhonatan Casale (jhc)
#
# Contact : jhonatan@jhonatancasale.com
#         : casale.jhon@gmail.com
#         : https://github.com/jhonatancasale
#         : https://twitter.com/jhonatancasale
#         : http://jhonatancasale.github.io/
#
# Create date Wed  8 Mar 13:32:21 BRT 2017
"""


from glob import glob as ls
from numpy import array, dot, random
import matplotlib.pyplot as plt


def f_activation(input_value, threshold=.5):
    """
    (double, double) -> int

    Calculate when the neuron fires
    """


    return 0 if input_value < threshold else 1


def build_dataset(filename):
    """
    str -> list

    TODO Update me!!!!!
    Return the content of `filename` in a numpy.matrix object
    """


    print("Building dataset from {}".format(filename))
    dataset = []
    with open(filename, "r") as _file:
        next(_file) # skip header
        for line in _file:
            values = [float(field) for field in line.split()]
            # add [1.0] to represent theta value
            dataset.append((array(values[:-1] + [1.0]), values[-1]))
        return dataset


def train(training_set, learning_rate=.1, max_iterations=1e4, error=1e-2):
    """
    (list [, double] [, int] [, int]) -> (list, list)

    Operate over the training set until the cost function produces an error
    lesser than the (optional) given param `error`. In this case, the function
    returns a tuple with `weights` adjusted values and a list containing (0,
    100] computed average errors per iteration

    Otherwise, return a tuple with None and the list with the first 100 average
    errors calculated
    """


    # training_set[0] is a tuple, training_set[0][0] is an array
    weights = array(random.random(training_set[0][0].size))

    squared_error_list = []
    for i in range(int(max_iterations)):
        squared_error = 0
        for sample, expected_output in training_set:
            diff = expected_output - f_activation(dot(sample, weights))
            squared_error += diff ** 2
            weights = weights - learning_rate * (2 * diff * -sample)
        squared_error /= len(training_set)
        if i < 100:
            squared_error_list.append(squared_error)
        if squared_error < error:
            return (weights, squared_error_list)
    return (None, squared_error_list)


def print_value_table(dataset, weights):
    """
    (list, list) -> None

    Prints on stdout a table containing both expected values and obtained
    values from each dataset entry

    Otherwise, just print `Convergence failure`
    """


    if weights is not None:
        print("Expected\tObtained")
        for sample, expected_output in dataset:
            print("{:<8}\t{}".
                  format(expected_output,
                         f_activation(dot(sample, weights)))
                 )
    else:
        print("Convergence failure")


def converged(dataset, weights):
    """
    (list, list) -> bool

    Checks if the expected output from each entry from the given param
    `dataset` are equal to the obtained output using the given param `weights`
    """


    if weights is None:
        return False
    for sample, expected_output in dataset:
        if expected_output != f_activation(dot(sample, weights)):
            return False
    return True


def plot_errors(errors):
    """
    (list) -> matplotlib.pyplot

    Produces a plot of the given param `errors`
    """


    plt.close()
    plt.xlabel("$Samples$")
    plt.ylabel("$Value$")
    plt.plot(errors, "b{}-".format("o" if len(errors) < 25 else ""),
             label="Squared Error in the first 100 samples")
    plt.legend(loc="upper right")
    plt.axis([-.1, len(errors), -.1, max(errors) + .5])
    plt.grid(True)
    plt.show()


def main():
    """
    Assumptions: input files *.dat on same dir
    For each *.dat file in the same dir, build a dataset for then and train
    the perceptron reporting the results
    """

    files = ls("*.dat")
    for case in files:
        dataset = build_dataset(case)

        weights, errors = train(dataset)
        print("Training convergence process: {}{}".
              format("Succeed" if converged(dataset, weights) else "Failed",
                     "\n" if case != files[-1] else ""
                    )
             )


        #TODO parse plot command line option
        if not converged(dataset, weights):
            plot_errors(errors)


if __name__ == '__main__':
    main()
