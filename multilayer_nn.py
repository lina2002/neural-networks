import autograd.numpy as np
import copy

from autograd import grad
from autograd.scipy.misc import logsumexp
from sklearn.utils.extmath import softmax


class MultiLayerNN:

    def __init__(self, sizes, batch_size, num_of_epochs, learning_rate, init_scale, keep_prob, ema):
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.ema = ema

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        self.weights = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            w = 2*init_scale*np.random.rand(s1, s2) - init_scale
            self.weights.append(w)

        self.ema_weights = copy.deepcopy(self.weights)

    def _feed(self, X, weights):
        z = np.dot(X, weights[0])
        for w in weights[1:]:
            a = relu(z)
            z = np.dot(a, w)
        return softmax(z)

    def _cost(self, X, y, weights):
        z = np.dot(X, weights[0])
        for w in weights[1:]:
            a = relu(z)
            a_d = dropout(a, self.keep_prob)
            z = np.dot(a_d, w)
        # print(np.log(self._feed(X)))
        # print(z - logsumexp(z, axis=1, keepdims=True))
        # print(y)
        # print(np.sum((z - logsumexp(z, axis=1, keepdims=True))*y, axis=1))
        return -np.sum(np.sum((z - logsumexp(z, axis=1, keepdims=True))*y, axis=1))/X.shape[0]

    def _d_cost(self, X, y, weights):
        return grad(self._cost, 2)(X, y, weights)

    def predict(self, X, weights=None):
        if weights is None:
            weights = self.ema_weights
        return np.argmax(self._feed(X, weights), 1)

    def fit(self, X, y, X_valid, y_valid):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], self.batch_size):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                delta_w = self._d_cost(X[selected_data_points], y[selected_data_points], self.weights)
                for w, d in zip(self.weights, delta_w):
                    w -= d*self.learning_rate
                for i in range(len(self.weights)):
                    self.ema_weights[i] = self.ema_weights[i]*self.ema + self.weights[i]*(1-self.ema)

            training_accuracy = compute_accuracy(self.predict(X, self.weights), np.argmax(y, 1))
            validation_accuracy = compute_accuracy(self.predict(X_valid, self.weights), np.argmax(y_valid, 1))

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(round(validation_accuracy, 2)))

            print("cost: " + str(self._cost(X, y, self.weights)))

            training_accuracy = compute_accuracy(self.predict(X, self.ema_weights), np.argmax(y, 1))
            validation_accuracy = compute_accuracy(self.predict(X_valid, self.ema_weights), np.argmax(y_valid, 1))

            print("training accuracy ema: " + str(round(training_accuracy, 2)))
            print("validation accuracy ema: " + str(round(validation_accuracy, 2)))


def compute_accuracy(predictions, labels):
    correctly_predicted = np.sum(predictions == labels)
    all = labels.shape[0]
    return 100*correctly_predicted/all


def relu(x):
    return np.maximum(x, 0)


def dropout(x, keep_prob):
    rand = np.random.rand(*x.shape)
    to_keep = rand < keep_prob
    return np.multiply(x, to_keep)/keep_prob

