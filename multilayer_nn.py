import autograd.numpy as np
import copy

from autograd import grad
from autograd.scipy.misc import logsumexp
from sklearn.utils.extmath import softmax

from utils import compute_accuracy


class MultiLayerNN:

    def __init__(self, sizes, batch_size, num_of_epochs, learning_rate, init_scale, keep_prob, ema):
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.ema = ema
        self.stds = []
        self.averages = []

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        self.weights = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            w = 2*init_scale*np.random.rand(s1, s2) - init_scale
            alpha = 2*init_scale*np.random.rand(s2) - init_scale
            beta = 2*init_scale*np.random.rand(s2) - init_scale
            self.weights.append(w)
            self.weights.append(alpha)
            self.weights.append(beta)
        self.weights = self.weights[:-2]

        self.ema_weights = copy.deepcopy(self.weights)

    def _feed(self, X, weights, test=False):
        z = np.dot(X, weights[0])
        if test:
            stds = self.stds
            averages = self.averages
        else:
            stds = weights[1::3]
            averages = weights[2::3]
        for alpha, beta, w in zip(stds, averages, weights[3::3]):
            a = batch_normalization(z, alpha, beta)
            a_r = relu(a)
            z = np.dot(a_r, w)
        return softmax(z)

    def _compute_stds_and_averages(self, X, weights):
        z = np.dot(X, weights[0])
        stds = weights[1::3]
        averages = weights[2::3]
        for alpha, beta, w in zip(stds, averages, weights[3::3]):
            a = self.saving_batch_normalization(z, alpha, beta)
            a_r = relu(a)
            z = np.dot(a_r, w)
        return softmax(z)

    def _cost(self, X, y, weights):
        z = np.dot(X, weights[0])
        for alpha, beta, w in zip(weights[1::3], weights[2::3], weights[3::3]):
            a = batch_normalization(z, alpha, beta)
            a_r = relu(a)
            a_d = dropout(a_r, self.keep_prob)
            z = np.dot(a_d, w)
        return -np.sum(np.sum((z - logsumexp(z, axis=1, keepdims=True))*y, axis=1))/X.shape[0]

    def _d_cost(self, X, y, weights):
        return grad(self._cost, 2)(X, y, weights)

    def predict(self, X, weights=None, test=False):
        if weights is None:
            weights = self.ema_weights
        return np.argmax(self._feed(X, weights, test), 1)

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

            self.save_average_and_std(X)

    def saving_batch_normalization(self, x, alpha, beta):
        epsilon = 0.000001
        self.averages.append(x.mean(axis=0))
        self.stds.append(x.std(axis=0))
        x_n = (x - x.mean(axis=0))/(x.std(axis=0) + epsilon)
        return np.multiply(x_n, (alpha + 1)) + beta

    def save_average_and_std(self, X):
        self._compute_stds_and_averages(X, self.ema_weights)


def batch_normalization(x, alpha, beta):
    epsilon = 0.000001
    x_n = (x - x.mean(axis=0))/(x.std(axis=0) + epsilon)
    return np.multiply(x_n, (alpha + 1)) + beta


def relu(x):
    return np.maximum(x, 0)


def dropout(x, keep_prob):
    rand = np.random.rand(*x.shape)
    to_keep = rand < keep_prob
    return np.multiply(x, to_keep)/keep_prob
