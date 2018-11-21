import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tqdm import tqdm


def compute_accuracy(predictions, labels):
    correctly_predicted = np.sum(predictions == labels)
    all = labels.shape[0]
    return 100*correctly_predicted/all


def train_validation_split(X, y, training_set_size):
    return X[:training_set_size], y[:training_set_size], X[training_set_size:], y[training_set_size:]


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


logs_path = "./logs/visualize_graph"


class MultiLayerNN:

    def __init__(self, batch_size, num_of_epochs, learning_rate, init_scale, keep_prob, ema):
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.keep_prob = keep_prob
        self.init_scale = init_scale

        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

        N = 128
        self.weights = [self.var([3, 3, 3, N]),
                        self.var([3, 3, N, 2*N]),
                        self.var([3, 3, 2*N, 2*N]),
                        self.var([3, 3, 2*N, 2*N]),
                        self.var([3, 3, 2*N, 2*N]),
                        self.var([3, 3, 2*N, 2*N]),
                        self.var([1, 1, 2*N, 10])]
        # self.ema_weights = [ema * ew + (1 - ema) * w for ew, w in zip(self.ema_weights, self.weights)]
        ema_sth = tf.train.ExponentialMovingAverage(ema)
        self.ema_op = ema_sth.apply(self.weights)
        self.ema_weights = [ema_sth.average(w) for w in self.weights]
        weights = self.WeightsGenerator()
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        X = tf.image.random_flip_left_right(X)
        angles = tf.random_normal([self.batch_size])
        angles = tf.nn.dropout(angles, 0.5)  # rotate only half of images
        self.X = tf.contrib.image.rotate(X, angles)
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.prob = tf.placeholder_with_default(1.0, shape=())
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

        d = tf.nn.dropout(m, self.prob)
        c = tf.nn.conv2d(d, next(weights), strides=[1, 1, 1, 1], padding="SAME")
        n = batch_norm(c, **self.bn_params)
        r = tf.nn.relu(n)

        z = tf.layers.average_pooling2d(r, pool_size=[8, 8], strides=[1, 1], padding="VALID")
        z = tf.squeeze(z)

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
        r = tf.add_n([r, to_add])
        return r

    # def predict(self, X):
    #     predictions = self.y_pred.eval({self.X: X})
    #     flipped_predictions = self.y_pred.eval({self.X: tf.image.flip_left_right(X).eval()})
    #     predicitons_max = np.max(predictions)
    #     flipped_predictions_max = np.max(flipped_predictions)
    #     if predicitons_max > flipped_predictions_max:
    #         return np.argmax(predictions, 1)
    #     else:
    #         return np.argmax(flipped_predictions, 1)

    def predict(self, X, e_weights=None):
        if e_weights:
            feed_dict = {w: we for w, we in zip(self.weights, e_weights)}
            feed_dict[self.X] = X
            return np.argmax(self.y_pred.eval(feed_dict), 1)
        return np.argmax(self.y_pred.eval({self.X: X}), 1)

    def fit(self, X_train, y_train, X_valid, y_valid):
        best_ema_validation_accuracy = 0

        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            permuted_indices = np.random.permutation(X_train.shape[0])
            for i in tqdm(range(0, X_train.shape[0], self.batch_size)):
                selected_data_points = np.take(permuted_indices, range(i, i+self.batch_size), mode='wrap')
                self.sess.run([self.optimizer, self.ema_op], {self.X: X_train[selected_data_points],
                                                              self.y: y_train[selected_data_points], self.prob: self.keep_prob,
                                                              self.is_training: True})

            training_predictions = self.get_predictions(X_train, False)
            training_accuracy = compute_accuracy(training_predictions, np.argmax(y_train, 1))

            validation_preditions = self.get_predictions(X_valid, False)
            validation_accuracy = compute_accuracy(validation_preditions, np.argmax(y_valid, 1))

            ema_validation_predictions = self.get_predictions(X_valid)
            ema_validation_accuracy = compute_accuracy(ema_validation_predictions, np.argmax(y_valid, 1))
            if ema_validation_accuracy > best_ema_validation_accuracy:
                best_ema_validation_accuracy = ema_validation_accuracy
                self.saver.save(sess=self.sess, save_path=self.save_dir + "best_model")

            print("training accuracy: " + str(round(training_accuracy, 2)))
            print("validation accuracy: " + str(round(validation_accuracy, 2)))
            print("ema validation accuracy: " + str(round(ema_validation_accuracy, 2)))

            summary = tf.Summary(value=[tf.Summary.Value(tag="training accuracy",
                                                         simple_value=training_accuracy)])
            self.writer.add_summary(summary, epoch)
            summary = tf.Summary(value=[tf.Summary.Value(tag="validation accuracy",
                                                         simple_value=validation_accuracy)])
            self.writer.add_summary(summary, epoch)

    def restore_the_best(self):
        self.saver.restore(sess=self.sess, save_path=self.save_dir + "best_model")

    def get_predictions(self, X, use_ema_weights=True):
        if use_ema_weights:
            e_weights = self.sess.run(self.ema_weights)
        else:
            e_weights = None
        preditions = []
        for i in range(0, X.shape[0], 100):
            preditions = np.append(preditions, self.predict(X[i:(i+100)], e_weights))
        return preditions

    def var(self, shape):
        return tf.Variable(tf.random_uniform(shape, minval=-self.init_scale, maxval=self.init_scale))

    def WeightsGenerator(self):
        for w in self.weights:
            yield w


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    y_train = dense_to_one_hot(y_train)
    y_test = np.squeeze(y_test)

    x_train, y_train, x_valid, y_valid = train_validation_split(x_train, y_train, training_set_size=45_000)

    params = {'batch_size': 32,
              'num_of_epochs': 30,
              'learning_rate': 0.1,
              'init_scale': 0.05,
              'keep_prob': 0.9,
              'ema': 0.999}
    model = MultiLayerNN(**params)
    model.fit(x_train, y_train, x_valid, y_valid)

    model.restore_the_best()
    print("RESTORED")
    predictions = model.get_predictions(x_test)

    print("test accuracy: " + str(round(compute_accuracy(predictions, y_test), 2)))
