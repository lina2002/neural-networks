import numpy as np
import tensorflow as tf
from tqdm import tqdm


training_file = '/Users/agnieszka.paszek/Documents/pan_tadeusz_1_10.txt'
validation_file = '/Users/agnieszka.paszek/Documents/pan_tadeusz_11.txt'
test_file = '/Users/agnieszka.paszek/Documents/pan_tadeusz_12.txt'


def read_data(filename):
    with open(filename, encoding='utf-8-sig', mode='U') as f:
        return f.read()


training_data = read_data(training_file)
print(len(training_data))
validation_data = read_data(validation_file)
test_data = read_data(test_file)
alphabet = sorted(
    set(training_data + validation_data + test_data))  # czy powinnam tutaj uzywac danych walidacyjnych i testowych,
# jesli nie to jakis domyslny indeks dla kazdego znaku, ktory nie wystepowal w treniningowym?
alphabet_size = len(alphabet)
print(alphabet_size)

index_to_char = {index: char for index, char in enumerate(alphabet)}
char_to_index = {char: index for index, char in enumerate(alphabet)}


def char_to_one_hot(char):
    index = char_to_index[char]
    one_hot = np.zeros(alphabet_size)
    one_hot[index] = 1
    return one_hot


def string_to_one_hots(string):
    return [char_to_one_hot(char) for char in string]


def indices_to_string(indices):
    return ''.join(index_to_char[i] for i in indices)


class LSTM:

    def __init__(self, x_size, y_size, num_of_epochs, learning_rate, init_scale, number_of_steps, batch_size, keep_prob):
        self.num_of_epochs = num_of_epochs
        self.h_size = 200
        self.x_size = x_size
        self.y_size = y_size
        self.batch_size = batch_size
        self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        self.init_scale = init_scale
        self.keep_prob = keep_prob

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.number_of_steps, self.x_size])
        self.targets = tf.placeholder(tf.float32, [self.batch_size, self.number_of_steps, self.y_size])
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.prob = tf.placeholder_with_default(1.0, shape=())

        self.W1 = tf.Variable(tf.random_uniform([1, self.x_size, self.h_size], minval=-init_scale, maxval=init_scale))
        W1_tiled = tf.tile(self.W1, (self.batch_size, 1, 1))
        cell_inputs = tf.matmul(self.inputs, W1_tiled)  # [batch_size, number_of_steps, h_size]

        # self.test = tf.Variable(tf.random_uniform([342543, 342], minval=-init_scale, maxval=init_scale))
        # tf.matmul(cell_inputs, self.test)

        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.h_size)
        outputs, states = tf.nn.dynamic_rnn(self.lstm_cell, cell_inputs, dtype=tf.float32)

        self.W2 = tf.Variable(tf.random_uniform([1, self.h_size, self.y_size], minval=-init_scale, maxval=init_scale))
        W2_tiled = tf.tile(self.W2, (self.batch_size, 1, 1))
        logits = tf.matmul(outputs, W2_tiled)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.targets))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def perplexity(self, data):
        pass

    def fit(self):
        training_data_encoded = np.array([char_to_one_hot(c) for c in list(training_data)])
        print(training_data_encoded.shape)
        training_data_2 = training_data_encoded
        l = len(training_data_2)
        l -= l % self.batch_size

        training_data_2 = training_data_2[:l]
        training_data_2 = np.array(list(training_data_2))
        training_data_2 = training_data_2.reshape((self.batch_size, -1, alphabet_size))

        # sanity check, na poczatku powinno byc ~wielkosci alfabetu
        print('validation perplexity:')
        print(self.perplexity(validation_data))
        best_validation_perplexity = 100
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            for i in tqdm(range((training_data_2.shape[1]-1)//self.number_of_steps)):
                inputs = training_data_2[:, i*self.number_of_steps:(i+1)*self.number_of_steps]
                targets = training_data_2[:, i*self.number_of_steps+1:(i+1)*self.number_of_steps+1]

                self.sess.run(self.optimizer, {self.inputs: inputs,
                                               self.targets: targets, self.prob: self.keep_prob,
                                               self.is_training: True})

            print('validation perplexity:')
            validation_perplexity = self.perplexity(validation_data)
            print(validation_perplexity)

            if validation_perplexity < best_validation_perplexity:
                best_validation_perplexity = validation_perplexity
            else:
                self.learning_rate /= 2

        print('test perplexity:')
        print(self.perplexity(test_data))


params = {'num_of_epochs': 35,
          'learning_rate': 4,
          'init_scale': 0.1,
          'number_of_steps': 25,
          'batch_size': 20,
          'keep_prob': 0.9}
lstm = LSTM(alphabet_size, alphabet_size, **params)
lstm.fit()
