import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tqdm import tqdm

from utils import compute_accuracy


logs_path = "./logs/visualize_graph"


class MultiLayerNN:

    def __init__(self, batch_size, num_of_epochs, learning_rate, init_scale, keep_prob, ema):
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.keep_prob = keep_prob

        N = 8
        weights = WeightsGenerator(N, init_scale)
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.prob = tf.placeholder_with_default(1.0, shape=())
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.bn_params = {
            'is_training': self.is_training,
            'decay': ema,
            'updates_collections': None
        }
        d = tf.nn.dropout(self.X, self.prob)
        c = tf.nn.conv2d(d, next(weights), strides=[1, 1, 1, 1], padding="SAME")  # [batch_size, 32, 32, N]
        n = batch_norm(c, **self.bn_params)
        r = tf.nn.relu(n)

        d = tf.nn.dropout(r, self.prob)
        c = tf.nn.conv2d(d, next(weights), strides=[1, 1, 1, 1], padding="SAME")  # [batch_size, 32, 32, 2N]
        n = batch_norm(c, **self.bn_params)
        r = tf.nn.relu(n)

        m = tf.layers.max_pooling2d(r, pool_size=[2, 2], strides=[2, 2], padding="SAME")

        r = self.residual(m, weights)
        r = self.residual(r, weights)

        m = tf.layers.max_pooling2d(r, pool_size=[2, 2], strides=[2, 2], padding="SAME")

        m = tf.reshape(m, (-1, 8*8*32*N))
        z = tf.matmul(m, next(weights))

        self.y_pred = tf.nn.softmax(z)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(logs_path, self.sess.graph)
        self.saver = tf.train.Saver()
        self.save_dir = 'saved_models/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def residual(self, r, weights):
        to_add = r
        d = tf.nn.dropout(r, self.prob)
        c = tf.nn.conv2d(d, next(weights), strides=[1, 1, 1, 1], padding="SAME")
        n = batch_norm(c, **self.bn_params)
        r = tf.nn.relu(n)

        d = tf.nn.dropout(r, self.prob)
        c = tf.nn.conv2d(d, next(weights), strides=[1, 1, 1, 1], padding="SAME")
        n = batch_norm(c, **self.bn_params)
        r = tf.nn.relu(n)

        to_add = tf.tile(to_add, [1, 1, 1, 4])
        r = tf.add_n([r, to_add])
        return r

    def predict(self, X):
        return np.argmax(self.y_pred.eval({self.X: X}), 1)

    def fit(self, X_train, y_train, X_valid, y_valid):
        best_validation_accuracy = 0

        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X_train.shape[0])
            for i in tqdm(range(0, X_train.shape[0], self.batch_size)):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                self.sess.run(self.optimizer, {self.X: X_train[selected_data_points],
                                               self.y: y_train[selected_data_points], self.prob: self.keep_prob,
                                               self.is_training: True})

            training_predictions = self.get_predictions(X_train)
            training_accuracy = compute_accuracy(training_predictions, np.argmax(y_train, 1))

            validation_preditions = self.get_predictions(X_valid)
            validation_accuracy = compute_accuracy(validation_preditions, np.argmax(y_valid, 1))
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                self.saver.save(sess=self.sess, save_path=self.save_dir + "best_model")

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(round(validation_accuracy, 2)))

            # summary = tf.Summary(value=[tf.Summary.Value(tag="training accuracy",
            #                                              simple_value=training_accuracy)])
            # self.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[tf.Summary.Value(tag="validation accuracy",
                                                         simple_value=validation_accuracy)])
            self.writer.add_summary(summary, epoch)

    def restore_the_best(self):
        self.saver.restore(sess=self.sess, save_path=self.save_dir + "best_model")

    def get_predictions(self, X):
        preditions = []
        for i in range(0, X.shape[0], self.batch_size):
            preditions = np.append(preditions, self.predict(X[i:(i+self.batch_size)]))
        return preditions


def WeightsGenerator(N, init_scale):
    yield tf.Variable(tf.random_uniform([3, 3, 3, N], minval=-init_scale, maxval=init_scale))
    yield tf.Variable(tf.random_uniform([3, 3, N, 2*N], minval=-init_scale, maxval=init_scale))
    yield tf.Variable(tf.random_uniform([3, 3, 2*N, 4*N], minval=-init_scale, maxval=init_scale))
    yield tf.Variable(tf.random_uniform([3, 3, 4*N, 8*N], minval=-init_scale, maxval=init_scale))
    yield tf.Variable(tf.random_uniform([3, 3, 8*N, 16*N], minval=-init_scale, maxval=init_scale))
    yield tf.Variable(tf.random_uniform([3, 3, 16*N, 32*N], minval=-init_scale, maxval=init_scale))
    yield tf.Variable(tf.random_uniform([8*8*32*N, 10], minval=-init_scale, maxval=init_scale))
