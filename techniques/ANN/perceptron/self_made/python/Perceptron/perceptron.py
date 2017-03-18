#!env python3
# -*- coding: utf-8 -*-

'''
# Author: Jhonatan Casale (jhc)
#
# Contact : jhonatan@jhonatancasale.com
#         : casale.jhon@gmail.com
#         : https://github.com/jhonatancasale
#         : https://twitter.com/jhonatancasale
#         : http://jhonatancasale.github.io/
#
# Create date Wed  8 Mar 13:32:21 BRT 2017
'''


from glob import glob as ls
from numpy import array, dot, random


def split_data(filename: str) -> (list, list, list, list):
    '''
    Split the content of `filename` in 2 datasets for training, and another
    2 datasets to test the predictions.
    '''


    print("Building dataset from {}".format(filename))
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []

    with open(filename, "r") as _file:
        next(_file) # skip header
        for line in _file:
            values = [float(field) for field in line.split()]
            # add [1.0] to represent theta value
            features_train.append(array(values[:-1] + [1.0]))
            labels_train.append(values[-1])
        features_test = features_train[:]
        labels_test = labels_train[:]
        return features_train, features_test, labels_train, labels_test


def print_value_table(labels_test: list, prediction: list) -> None:
    '''
    Prints on the screen the `expected value` followed by the `obtained value`
    '''


    print("Expected\tObtained")
    for label, pred in zip(labels_test, prediction):
        print("{:<8}\t{}".format(label, pred))


def test_case(case: str) -> float:
    '''
    Open the file with the given name `case` and parse, split into datasets and
    train the perceptron.

    Returns the value of the accuracy obtained after the training process
    '''


    clf = Perceptron()

    features_train, features_test, labels_train, labels_test = split_data(case)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    return accuracy_score(labels_test, pred), clf


def main():
    '''
    Assumptions: input files *.dat on same dir
    For each *.dat file in the same dir, build a dataset for then and train
    the perceptron reporting the accuracy score obtained
    '''

    for case in ls("*.dat"):
        accuracy, clf = test_case(case)
        print('Convergence: {}\nAccuracy: {}%\n'.
              format("Succeed" if clf.converged else "Fail", 100 * accuracy))
        print(clf.weights)


###############################################################################


def accuracy_score(labels_test: list, prediction: list) -> float:
    '''
    Return the calculated accuracy score, in other words, the number of `right`
    predictions divided by `all` predictions
    '''


    if len(labels_test) != len(prediction):
        raise ValueError('Both `labels_test` and `predictions` must have \
                         the same length')

    test_samples = zip(prediction, labels_test)
    return sum([1 for p, l in  test_samples if p == l]) / len(labels_test)


class Perceptron(object):
    '''
    Implements the single neuron (perceptron) Artificial Neural Network (ANN)
    '''


    def __init__(self: object):
        '''
        Naive initializations
        '''


        self.weights = []
        self.errors = []
        self.converged = False


    def fit(self: object, features_train: list, labels_train: list,
            learning_rate=.1, max_iterations=int(1e4), error=1e-2) -> None:
        '''
        Operate over the training set until the cost function produces a square
        error lesser than the (optional) given param `error`. Or until hit the
        (optional) given param `max_iterations`
        '''


        self.converged = False
        self.errors = []
        self.weights = array(random.random(features_train[0].size))

        for iteration in range(max_iterations):
            squared_error = 0
            for features, label in zip(features_train, labels_train):
                diff = label - self.net_output(features)
                squared_error += diff ** 2
                self.weights = self.weights - learning_rate * (2 * diff * -features)
            if iteration < 100:
                self.errors.append(squared_error)
            if squared_error / len(features_train) < error:
                self.converged = True
                break


    def predict(self: object, features_test: list) -> list:
        '''
        Make a prediction to the given param `features_test`
        '''


        return [self.net_output(features) for features in features_test]


    def net_output(self: object, feature_sample: list) -> float:
        '''
        Calculate the output of the ANN when presented with the values of the
        given param `feature_sample`


        Returns: The result of the activation function with the pondered
        weights applied to the values of `feature_sample`
        '''


        return self.f_activation(dot(feature_sample, self.weights))


    def plot_errors(self: object) -> None:
        '''
        Generate a plot with the error curve obtained during the last training
        of the perceptron
        '''


        import matplotlib.pyplot as plt


        plt.close()
        plt.xlabel("$Samples$")
        plt.ylabel("$Value$")
        plt.plot(
            self.errors, "b{}-".format("o" if len(self.errors) < 25 else ""),
            label="Squared Error in the first {} samples".
            format(len(self.errors))
        )
        plt.legend(loc="upper right")
        plt.axis([-.1, len(self.errors), -.1, max(self.errors) + .5])
        plt.grid(True)
        plt.show()


    def f_activation(self: object, input_value: float, threshold=.5) -> int:
        '''
        Calculate when the neuron fires.

        Returns
            1 - If the given param `input_value` is greater or equal than the
                (optional given param `threshold`.
            0 - Otherwise.
        '''


        return 0 if input_value < threshold else 1


if __name__ == '__main__':
    main()
