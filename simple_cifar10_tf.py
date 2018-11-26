import numpy as np
import tensorflow as tf

from extract_data import dense_to_one_hot
from simple_nn_tf import SimpleNN
from utils import train_validation_split, compute_accuracy

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    y_train = dense_to_one_hot(y_train)
    y_test = np.squeeze(y_test)
    x_train = np.reshape(x_train, (-1, 32*32*3))
    x_test = np.reshape(x_test, (-1, 32*32*3))

    x_train, y_train, x_valid, y_valid = train_validation_split(x_train, y_train, training_set_size=45_000)

    params = {'batch_size': 32,
              'num_of_epochs': 5,
              'learning_rate': 0.1,
              'init_scale': 0.05,
              'keep_prob': 0.9,
              'ema': 0.999}
    model = SimpleNN([32*32*3, 500, 10], **params)
    model.fit(x_train, y_train, x_valid, y_valid)

    predictions = model.predict(x_test)

    print("test accuracy: " + str(round(compute_accuracy(predictions, y_test), 2)))
