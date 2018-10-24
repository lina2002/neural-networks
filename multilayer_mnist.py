import autograd.numpy as np

from extract_data import extract_images, extract_labels
from multilayer_nn import compute_accuracy, MultiLayerNN
from plotting import plot_confusion_matrices

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)


def shuffle(X, y):
    randomize = np.random.permutation(X.shape[0])
    return X[randomize], y[randomize]


def train_validation_split(X, y, training_set_size):
    return X[:training_set_size], y[:training_set_size], X[training_set_size:], y[training_set_size:]


if __name__ == "__main__":
    images = extract_images('train-images-idx3-ubyte.gz')
    images = np.reshape(images, (-1, 28*28))/255
    labels = extract_labels('train-labels-idx1-ubyte.gz', one_hot=True)

    images, labels = shuffle(images, labels)

    training_set_size = 55_000
    training_images, training_labels, valid_images, valid_labels \
        = train_validation_split(images, labels, training_set_size)

    model = MultiLayerNN(28*28, 10, 500, 0.9)
    model.fit(training_images, training_labels, valid_images, valid_labels)

    eval_images = extract_images('t10k-images-idx3-ubyte.gz')
    eval_images = np.reshape(eval_images, (-1, 28*28))/255
    eval_labels = extract_labels('t10k-labels-idx1-ubyte.gz')

    predictions = model.predict(eval_images)

    print(compute_accuracy(predictions, eval_labels))

    plot_confusion_matrices(eval_labels, predictions, classes=range(10))
