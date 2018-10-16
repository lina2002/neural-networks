import gzip
import numpy as np


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


def sigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x))


class SingleLayerNN:
    init_scale = 0.05
    learning_rate = 0.1
    batch_size = 128
    num_of_epochs = 10
    # num_of_epochs = 1
    # batch_size = 12

    def __init__(self, input_size, output_size, number_of_data_points):
        self.input_size = input_size
        self.output_size = output_size
        self.number_of_data_points = number_of_data_points

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        self.weights = 2 * self.init_scale * np.random.rand(input_size, output_size) - self.init_scale

    def _feed(self, X):
        return sigmoid(np.dot(X, self.weights))

    def _cost(self, X, y):
        y_pred = self._feed(X)
        return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred)) # !!!!!

    def fit(self, X, y):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(self.number_of_data_points)
            for i in range(0, self.number_of_data_points, self.batch_size):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                X[selected_data_points]


if __name__ == "__main__":
    images = extract_images('train-images-idx3-ubyte.gz')
    print(type(images))
    images = np.reshape(images, (-1, 28*28))
    print(images.shape)
    labels = extract_labels('train-labels-idx1-ubyte.gz', one_hot=True)
    print(type(labels))
    print(labels.shape)

    training_images = images[0:55_000]
    print(training_images.shape)
    training_labels = labels[0:55_000]

    model = SingleLayerNN(28*28, 10, 55_000)
    model.fit(training_images, training_labels)
