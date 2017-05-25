#!env python3
# -*- coding: utf-8 -*-

from MLP import MLP
import pandas as pd


def load_dataset(filename: str) -> pd.DataFrame:
    """Load and return the content of `filename` into a Pandas DataFrame"""


    return pd.read_csv(filename, delim_whitespace=True)


def split_dataset(dataset: pd.DataFrame) -> (list, list, list, list):
    """
    Split the `dataset` into 4 lists: 2 for training and 2 to test the
    prediction
    """


    labels_train = dataset.values[:, -1]
    training_samples = dataset.values[:, :-1]

    return training_samples, training_samples[:], labels_train, labels_train[:]


def main():
    """
    Run the test case XOR
    """


    print("...Reading dataset")
    dataset = load_dataset("datasets/xor.dat")
    print("...done!")

    print("...Spliting the dataset")
    training_samples, testing_samples, labels_train, labels_test = split_dataset(dataset)
    print("...done!")

    print("...Creating the classifier")
    clf = MLP(input_layer=2, hidden=2, output=1)
    print("...done!")

    print("...Fiting the clf")
    clf.fit(training_samples, labels_train, verbose_error=True)
    print("...done!")

    print("...Made a prediction")
    pred = clf.predict(testing_samples)
    print("...done!")

    print('Convergence: with MSE:{}'.format(clf.error))

    print(clf)

    print(pd.DataFrame.from_items([('Expected', labels_test), ('Obtained', pred)]))

    clf.plot_errors()


if __name__ == '__main__':
    main()
