import autograd.numpy as np

from autograd import grad
from autograd.scipy.misc import logsumexp
from extract_data import extract_images, extract_labels
from plotting import plot_confusion_matrices
from sklearn.utils.extmath import softmax
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)


class SingleLayerNN:
    init_scale = 0.05
    learning_rate = 0.1
    batch_size = 10
    num_of_epochs = 20

    def __init__(self, number_of_inputs, number_of_outputs):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        self.weights = 2*self.init_scale*np.random.rand(number_of_inputs, number_of_outputs) - self.init_scale

        self.validation_accuracy = 0
        self.old_weights = self.weights

    def _feed(self, X):
        return softmax(np.dot(X, self.weights))

    def _cost(self, X, y, weights):
        z = np.dot(X, weights)
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
                self.weights -= delta_w * self.learning_rate

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
                self.learning_rate = 0.5*self.learning_rate


def compute_accuracy(predictions, labels):
    correctly_predicted = np.sum(predictions == labels)
    all = labels.shape[0]
    return 100*correctly_predicted/all


def train_validation_split(X, y, training_set_size):
    return X[:training_set_size], y[:training_set_size], X[training_set_size:], y[training_set_size:]


def shuffle(X, y):
    randomize = np.random.permutation(X.shape[0])
    return X[randomize], y[randomize]


if __name__ == "__main__":
    images = extract_images('train-images-idx3-ubyte.gz')
    images = np.reshape(images, (-1, 28*28))/255
    labels = extract_labels('train-labels-idx1-ubyte.gz', one_hot=True)

    images, labels = shuffle(images, labels)

    training_set_size = 55_000
    training_images, training_labels, valid_images, valid_labels \
        = train_validation_split(images, labels, training_set_size)

    model = SingleLayerNN(28*28, 10)
    model.fit(training_images, training_labels, valid_images, valid_labels)

    eval_images = extract_images('t10k-images-idx3-ubyte.gz')
    eval_images = np.reshape(eval_images, (-1, 28*28))/255
    eval_labels = extract_labels('t10k-labels-idx1-ubyte.gz')

    predictions = model.predict(eval_images)

    print(compute_accuracy(predictions, eval_labels))

    plot_confusion_matrices(eval_labels, predictions, classes=range(10))
