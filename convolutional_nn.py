from autograd import numpy as np, grad
from autograd.scipy.misc import logsumexp
from autograd.scipy.signal import convolve
from sklearn.utils.extmath import softmax
from tqdm import tqdm

from utils import compute_accuracy


class ConvolutionalNN:
    def __init__(self, output_size, batch_size, num_of_epochs, learning_rate, init_scale, keep_prob):
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.init_scale = init_scale
        self.keep_prob = keep_prob

        N = 8
        self.weights = []
        self.weights.append(2*init_scale*np.random.rand(N, 3, 3, 3) - init_scale)
        self.weights.append(2*init_scale*np.random.rand(2*N, N, 4, 4) - init_scale)
        self.weights.append(2*init_scale*np.random.rand(2*N*6*6, 10) - init_scale)
        self.weights.append(2*init_scale*np.random.rand(1, N, 1, 1) - init_scale)
        self.weights.append(2*init_scale*np.random.rand(1, N, 1, 1) - init_scale)
        self.weights.append(2*init_scale*np.random.rand(1, 2*N, 1, 1) - init_scale)
        self.weights.append(2*init_scale*np.random.rand(1, 2*N, 1, 1) - init_scale)

    def _feed(self, X, weights):
        c = convolve(X, weights[0], mode='valid', axes=([1, 2], [2, 3]), dot_axes=([3], [1]))
        n = batch_normalization(c, weights[3], weights[4])
        r = relu(n)
        m = maxout(r)
        c = convolve(m, weights[1], mode='valid', axes=([2, 3], [2, 3]), dot_axes=([1], [1]))
        n = batch_normalization(c, weights[5], weights[6])
        r = relu(n)
        m = maxout(r)
        m = np.reshape(m, (X.shape[0], -1))
        z = np.dot(m, weights[2])
        return softmax(z)

    def predict(self, X):
        return np.argmax(self._feed(X, self.weights), 1)

    def _cost(self, X, y, weights):
        d = dropout(X, self.keep_prob)
        c = convolve(d, weights[0], mode='valid', axes=([1, 2], [2, 3]), dot_axes=([3], [1])) # [batch_size, N, 30, 30]
        n = batch_normalization(c, weights[3], weights[4])
        r = relu(n)
        m = maxout(r)
        d = dropout(m, self.keep_prob)
        c = convolve(d, weights[1], mode='valid', axes=([2, 3], [2, 3]), dot_axes=([1], [1])) # [batch_size, 2N, 12, 12]
        n = batch_normalization(c, weights[5], weights[6])
        r = relu(n)
        m = maxout(r)
        m = np.reshape(m, (X.shape[0], -1))
        d = dropout(m, self.keep_prob)
        z = np.dot(d, weights[2])
        return -np.sum(np.sum((z - logsumexp(z, axis=1, keepdims=True))*y, axis=1))/X.shape[0]

    def _d_cost(self, X, y, weights):
        return grad(self._cost, 2)(X, y, weights)

    def fit(self, X, y, X_valid, y_valid):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X.shape[0])
            for i in tqdm(range(0, X.shape[0], self.batch_size)):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                delta_w = self._d_cost(X[selected_data_points], y[selected_data_points], self.weights)
                for w, d in zip(self.weights, delta_w):
                    w -= d*self.learning_rate

            training_accuracy = compute_accuracy(self.predict(X), np.argmax(y, 1))
            validation_accuracy = compute_accuracy(self.predict(X_valid), np.argmax(y_valid, 1))

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(round(validation_accuracy, 2)))

            # print("cost: " + str(self._cost(X, y, self.weights)))


def maxout(x):
    y = np.reshape(x, [x.shape[0], x.shape[1], x.shape[2]//2, 2, x.shape[3]//2, 2])
    return np.max(y, axis=(3, 5))


def relu(x):
    return np.maximum(x, 0)


def dropout(x, keep_prob):
    rand = np.random.rand(*x.shape)
    to_keep = rand < keep_prob
    return np.multiply(x, to_keep)/keep_prob


def batch_normalization(x, alpha, beta):
    epsilon = 0.000001
    x_n = (x - x.mean(axis=(0, 2, 3), keepdims=True))/(x.std(axis=(0, 2, 3), keepdims=True) + epsilon)
    return np.multiply(x_n, (alpha + 1)) + beta
