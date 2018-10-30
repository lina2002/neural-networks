import skimage
from autograd import numpy as np, grad
from autograd.scipy.misc import logsumexp
from autograd.scipy.signal import convolve


class ConvolutionalNN:
    def __init__(self, output_size, batch_size, num_of_epochs, learning_rate, init_scale):
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.init_scale = init_scale

        N = 8
        self.weights = []
        self.weights.append(2*init_scale*np.random.rand(N, 3, 3) - init_scale)
        self.weights.append(2*init_scale*np.random.rand(8*13*13, 10) - init_scale)

    def _cost(self, X, y, weights):
        c = convolve(X, weights[0], mode='valid', axes=([1, 2], [1, 2]))
        m = np.zeros([c.shape[0], c.shape[1], c.shape[2]//2, c.shape[3]//2])
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                m[i][j] = maxout(c[i][j])
        m = np.reshape(m, (self.batch_size, -1))
        z = np.dot(m, weights[1])
        return -np.sum(np.sum((z - logsumexp(z, axis=1, keepdims=True))*y, axis=1))/X.shape[0]

    def _d_cost(self, X, y, weights):
        return grad(self._cost, 2)(X, y, weights)

    def fit(self, X, y, X_valid, y_valid):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], self.batch_size):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                self._cost(X[selected_data_points], y[selected_data_points], self.weights)


def maxout(a):
    return skimage.measure.block_reduce(a, (2,2), np.max)