import numpy as np
import tensorflow as tf

from utils import compute_accuracy


class MultiLayerNN:

    def __init__(self, sizes, batch_size, num_of_epochs, learning_rate, init_scale, keep_prob):
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        weights = []
        # for s1, s2 in zip(sizes[:-1], sizes[1:]):
        #     w = tf.Variable(tf.random_uniform([s1, s2], minval=-init_scale, maxval=init_scale))
        #     weights.append(w)
        w = tf.Variable(tf.random_uniform([784, 10], minval=-init_scale, maxval=init_scale))
        weights.append(w)

        self.X = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        z = tf.matmul(self.X, weights[0])
        # for w in weights[1:]:
        #     z_r = tf.nn.relu(z)
        #     z = tf.matmul(z_r, w)
        self.y_pred = tf.nn.softmax(z)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.y)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, xs):
        return np.argmax(self.y_pred.eval({self.X: xs}), 1)

    def fit(self, X_train, y_train, X_valid, y_valid):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X_train.shape[0])
            for i in range(0, X_train.shape[0], self.batch_size):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                self.sess.run(self.optimizer, {self.X: X_train[selected_data_points], self.y: y_train[selected_data_points]})

            training_accuracy = compute_accuracy(self.predict(X_train), np.argmax(y_train, 1))
            validation_accuracy = compute_accuracy(self.predict(X_valid), np.argmax(y_valid, 1))

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(round(validation_accuracy, 2)))
