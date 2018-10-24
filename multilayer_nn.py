import autograd.numpy as np

from autograd import grad
from autograd.scipy.misc import logsumexp
from sklearn.utils.extmath import softmax


class MultiLayerNN:
    init_scale = 0.05
    learning_rate = 0.1
    batch_size = 64
    num_of_epochs = 20

    def __init__(self, number_of_inputs, number_of_outputs, hidden_layer_size, keep_prob):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.hidden_layer_size = hidden_layer_size
        self.keep_prob = keep_prob

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        w0 = 2*self.init_scale*np.random.rand(number_of_inputs, hidden_layer_size) - self.init_scale
        w1 = 2*self.init_scale*np.random.rand(hidden_layer_size, number_of_outputs) - self.init_scale
        self.weights = [w0, w1]

        self.validation_accuracy = 0
        self.old_weights = self.weights

    def _feed(self, X):
        z1 = np.dot(X, self.weights[0])
        a1 = relu(z1)
        z = np.dot(a1, self.weights[1])
        return softmax(z)

    def _cost(self, X, y, weights):
        z1 = np.dot(X, weights[0])
        a1 = relu(z1)
        a1_d = dropout(a1, self.keep_prob)
        z = np.dot(a1_d, weights[1])
        # print(np.log(self._feed(X)))
        # print(z - logsumexp(z, axis=1, keepdims=True))
        # print(y)
        # print(np.sum((z - logsumexp(z, axis=1, keepdims=True))*y, axis=1))
        return -np.sum(np.sum((z - logsumexp(z, axis=1, keepdims=True))*y, axis=1))/X.shape[0]

    def _d_cost(self, X, y, weights):
        return grad(self._cost, 2)(X, y, weights)

    def predict(self, X):
        return np.argmax(self._feed(X), 1)

    def fit(self, X, y, X_valid, y_valid):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], self.batch_size):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                delta_w = self._d_cost(X[selected_data_points], y[selected_data_points], self.weights)
                for w, d in zip(self.weights, delta_w):
                    w -= d*self.learning_rate

            training_accuracy = compute_accuracy(self.predict(X), np.argmax(y, 1))
            validation_accuracy = compute_accuracy(self.predict(X_valid), np.argmax(y_valid, 1))

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(validation_accuracy))

            print("cost: " + str(self._cost(X, y, self.weights)))

            if self.validation_accuracy < validation_accuracy:
                self.validation_accuracy = validation_accuracy
                self.old_weights = self.weights
            else:
                self.weights = self.old_weights
            #     self.learning_rate = 0.5*self.learning_rate


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

