import gzip
import numpy as np
from sklearn.utils.extmath import softmax
from scipy.misc import logsumexp


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class SingleLayerNN:
    init_scale = 0.05
    learning_rate = 0.1
    batch_size = 128
    num_of_epochs = 10

    def __init__(self, number_of_inputs, number_of_outputs, number_of_data_points):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.number_of_data_points = number_of_data_points

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        self.weights = 2 * self.init_scale * np.random.rand(number_of_inputs, number_of_outputs) - self.init_scale

    def _feed(self, X):
        return softmax(np.dot(X, self.weights))

    def _cost(self, X, y): # think about it!
        y_pred = self._feed(X)
        return -np.mean(np.dot(y, np.log(y_pred)))

    def _d_cost(self, X, y):
        y_pred = self._feed(X)
        return np.dot(np.transpose(X), y_pred-y)

    def predict(self, X):
            return np.argmax(self._feed(X), 1)

    def fit(self, X, y, X_valid, y_valid):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(self.number_of_data_points)
            for i in range(0, self.number_of_data_points, self.batch_size):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                delta = self._d_cost(X[selected_data_points], y[selected_data_points])
                self.weights -= delta * self.learning_rate

            training_accuracy = compute_accuracy(self.predict(X), np.argmax(y, 1))
            validation_accuracy = compute_accuracy(self.predict(X_valid), np.argmax(y_valid, 1))

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(validation_accuracy))


def compute_accuracy(predictions, labels):
    correctly_predicted = np.sum(predictions==labels)
    all = labels.shape[0]
    return 100*correctly_predicted/all


if __name__ == "__main__":
    images = extract_images('train-images-idx3-ubyte.gz')
    images = np.reshape(images, (-1, 28*28))
    labels = extract_labels('train-labels-idx1-ubyte.gz', one_hot=True)

    train_data_size = 55_000
    training_images = images[:train_data_size]
    training_labels = labels[:train_data_size]
    valid_images = images[train_data_size:]
    valid_labels = labels[train_data_size:]

    model = SingleLayerNN(28*28, 10, train_data_size)
    model.fit(training_images, training_labels, valid_images, valid_labels)

    eval_images = extract_images('t10k-images-idx3-ubyte.gz')
    eval_images = np.reshape(eval_images, (-1, 28*28))
    eval_labels = extract_labels('t10k-labels-idx1-ubyte.gz')
    predictions = model.predict(eval_images)

    print(compute_accuracy(predictions, eval_labels))

