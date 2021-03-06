import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from utils import compute_accuracy


logs_path = "./logs/visualize_graph"


class SimpleNN:

    def __init__(self, sizes, batch_size, num_of_epochs, learning_rate, init_scale, keep_prob, ema):
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.keep_prob = keep_prob

        # this will populate weights using numbers from range [-init_scale, init_scale) with uniform distribution
        weights = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            w = tf.Variable(tf.random_uniform([s1, s2], minval=-init_scale, maxval=init_scale))
            weights.append(w)

        self.X = tf.placeholder(tf.float32, [None, sizes[0]])
        self.y = tf.placeholder(tf.float32, [None, sizes[-1]])
        self.prob = tf.placeholder_with_default(1.0, shape=())
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.bn_params = {
            'is_training': self.is_training,
            'decay': ema,
            'updates_collections': None
        }
        z = tf.matmul(self.X, weights[0])
        for w in weights[1:]:
            z_n = batch_norm(z, **self.bn_params)
            z_r = tf.nn.relu(z_n)
            z_d = tf.nn.dropout(z_r, self.prob)
            z = tf.matmul(z_d, w)
        self.y_pred = tf.nn.softmax(z)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(logs_path, self.sess.graph)
        self.image = tf.summary.image('some images', tf.reshape(self.X, [-1, 32, 32, 3]), max_outputs=3)

    def predict(self, X):
        probabilities = self.y_pred.eval({self.X: X})
        return np.argmax(probabilities, 1)

    def fit(self, X_train, y_train, X_valid, y_valid):
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X_train.shape[0])
            for i in range(0, X_train.shape[0], self.batch_size):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                _, image = self.sess.run([self.optimizer, self.image], {self.X: X_train[selected_data_points],
                                               self.y: y_train[selected_data_points], self.prob: self.keep_prob,
                                               self.is_training: True})

            training_accuracy = compute_accuracy(self.predict(X_train), np.argmax(y_train, 1))
            validation_accuracy = compute_accuracy(self.predict(X_valid), np.argmax(y_valid, 1))

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(round(validation_accuracy, 2)))

            summary = tf.Summary(value=[tf.Summary.Value(tag="training accuracy",
                                                         simple_value=training_accuracy)])
            self.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[tf.Summary.Value(tag="validation accuracy",
                                                         simple_value=validation_accuracy)])
            self.writer.add_summary(summary, epoch)

            self.writer.add_summary(image, epoch)
