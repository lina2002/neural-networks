from autograd import numpy as np


def shuffle(X, y):
    randomize = np.random.permutation(X.shape[0])
    return X[randomize], y[randomize]


def train_validation_split(X, y, training_set_size):
    return X[:training_set_size], y[:training_set_size], X[training_set_size:], y[training_set_size:]


def compute_accuracy(predictions, labels):
    correctly_predicted = np.sum(predictions == labels)
    all = labels.shape[0]
    return 100*correctly_predicted/all
