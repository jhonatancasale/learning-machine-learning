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
# Create date Fri 24 Mar 07:11:54 -03 2017
'''

from math import exp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MLP(object):
    '''Implements a Multilayer Percemptron (MLP) Artificial Neural Network (ANN)'''


    def __init__(self: object, *, input_layer: int, hidden: int, output: int):
        '''
        '''

        self.converged = False

        self.layers = {}
        self.layers['hidden'] = self.create_layer(hidden, input_layer + 1)
        self.layers['output'] = self.create_layer(output, hidden + 1)

        self.hidden_shape = (hidden, input_layer + 1)
        self.output_shape = (output, hidden + 1)

    def create_layer(self: object, rows: int, columns: int) -> np.array:
        '''
        '''


        #np.random.random -> [0.0, 1.0) with - .5 --> [-0.5, 0.5)
        return np.random.random(rows * columns).reshape(rows, columns) - .5

    def forward(self: object, pattern: list) -> dict:
        '''
        '''


        forward_values = {}
        for key in ['hidden', 'hidden_df', 'output', 'output_df']:
            func = self.df_dnet if key.endswith('_df') else self.neuron_output
            if 'output' in key:
                #output layer receives hidden layer output + theta
                pattern = np.concatenate((forward_values['hidden'], [1.]), axis=0)
            layer = self.layers[key.strip('_df')]
            forward_values[key] = [func(np.dot(pattern, neuron)) for neuron in layer]

        return forward_values


    def fit(self: object, training_samples: list, labels_train: list,
            learning_rate=.1, max_iterations=int(1e4), error=1e-2) -> None:
        '''
        Perform Backpropagation algorithm until the cost function produces a
        square error lesser than the (optional) given param `error`. Or until
        hit the (optional) given param `max_iterations`
        '''


        #TODO
        pass


    def predict(self: object, test_samples: list) -> list:
        '''
        Make a prediction to the given param `test_samples`
        '''


        return [self.neuron_output(sample) for sample in test_samples]


    def df_dnet(self: object, net: int) -> float:
        '''Returns f(net) * (1 - f(net))'''


        _ = self.neuron_output(net)
        return _ * (1 - _)

    def neuron_output(self: object, net: int) -> float:
        '''
        '''

        return 1 / (1 + exp(-net))

    @property
    def weights(self: object) -> np.array:
        return self.layers

    def show_weights(self: object) -> None:
        print("Hidden weights\n{}".format(self.layers['hidden']))
        print("Output weights\n{}".format(self.layers['output']))


    def plot_errors(self: object) -> None:
        '''
        Generate a plot with the error curve obtained during the last training
        of the MLP
        '''


        plt.xlabel("$Iterations$")
        plt.ylabel("$Error - (MSE)$")
        plt.plot(
            self.errors, "b{}-".format("o" if len(self.errors) < 25 else ""),
            label="Mean Square Error (MSE) in the first {} iterations".
            format(len(self.errors))
        )
        plt.legend(loc="upper right")
        plt.axis([-.1, len(self.errors), -.1, max(self.errors) + .5])
        plt.grid(True)
        plt.show()



def main():
    '''
    '''
    clf = MLP(input_layer=2, hidden=2, output=1)
    #clf.show_weights()
    print(clf.forward([0, 0, 1]))

if __name__ == '__main__':
    main()
