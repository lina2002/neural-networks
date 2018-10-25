import autograd.numpy as np
import tensorflow as tf

from extract_data import dense_to_one_hot
from multilayer_nn import MultiLayerNN, compute_accuracy
from plotting import plot_confusion_matrices
from utils import train_validation_split

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = np.reshape(x_train, (-1, 32*32*3))/255
    x_test = np.reshape(x_test, (-1, 32*32*3))/255
    y_train = dense_to_one_hot(y_train)
    y_test = np.reshape(y_test, 10000)

    x_train, y_train, x_valid, y_valid = train_validation_split(x_train, y_train, training_set_size=45_000)

    model = MultiLayerNN([32*32*3, 500, 10], 0.9)
    model.fit(x_train, y_train, x_valid, y_valid)

    predictions = model.predict(x_test)

    print("test accuracy: " + str(round(compute_accuracy(predictions, y_test), 2)))

    plot_confusion_matrices(y_test, predictions, classes=range(10))
