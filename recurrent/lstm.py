import autograd.numpy as np
from autograd import grad
from autograd.scipy.misc import logsumexp
from tqdm import tqdm

from utils import sigmoid


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


def get_dropout_function(shape, keep_prob):
    mask = get_dropout_mask(shape, keep_prob)

    def dropout(x):
        return np.multiply(x, mask)
    return dropout


def get_dropout_mask(shape, keep_prob):
    rand = np.random.rand(*shape)
    to_keep = rand < keep_prob
    return to_keep


class LSTM:

    def __init__(self, x_size, y_size, num_of_epochs, learning_rate, init_scale, number_of_steps, batch_size, keep_prob):
        self.num_of_epochs = num_of_epochs
        self.h_size = h_size = 200
        self.x_size = x_size
        self.y_size = y_size
        self.batch_size = batch_size
        self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        self.init_scale = init_scale
        self.keep_prob = keep_prob
        self.C = np.zeros(h_size)
        self.h = np.zeros(h_size)
        self.W_1 = self._random_matrix((h_size, x_size))
        self.b_1 = self._random_matrix((h_size,))
        self.W_f = self._random_matrix((h_size, h_size + h_size))
        self.b_f = self._random_matrix((h_size,)) + 1
        self.W_i = self._random_matrix((h_size, h_size + h_size))
        self.b_i = self._random_matrix((h_size,))
        self.W_c = self._random_matrix((h_size, h_size + h_size))
        self.b_c = self._random_matrix((h_size,))
        self.W_o = self._random_matrix((h_size, h_size + h_size))
        self.b_o = self._random_matrix((h_size,))
        self.W_2 = self._random_matrix((y_size, h_size))
        self.b_2 = self._random_matrix((y_size,))
        self.weights = [self.W_1, self. b_1, self.W_f, self.b_f, self.W_i, self.b_i, self.W_c, self.b_c, self.W_o, self.b_o, self.W_2, self.b_2]

    def _cost(self, inputs, targets, hprev, Cprev, weights, disable_tqdm=True):
        W_1, b_1, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W_2, b_2 = weights
        h = np.copy(hprev)
        C = np.copy(Cprev)

        loss = 0
        for t in tqdm(range(len(inputs)), disable=disable_tqdm):
            x = char_to_one_hot(inputs[t])

            x = np.matmul(W_1, x) + b_1

            f = sigmoid(np.matmul(W_f, np.concatenate((h, x))) + b_f)
            i = sigmoid(np.matmul(W_i, np.concatenate((h, x))) + b_i)
            C_hat = np.tanh(np.matmul(W_c, np.concatenate((h, x))) + b_c)
            C = f*C + i*C_hat
            o = sigmoid(np.matmul(W_o, np.concatenate((h, x))) + b_o)
            h = o*np.tanh(C)
            y = np.matmul(W_2, h) + b_2

            target_index = char_to_index[targets[t]]
            # ps_target[t] = np.exp(ys[t][target_index])/np.sum(np.exp(ys[t]))  # probability for next chars being target
            # loss += -np.log(ps_target[t])
            loss += -(y[target_index] - logsumexp(y))

        loss = loss/len(inputs)
        return loss

    def _cost_batched(self, inputs, targets, hprev, Cprev, weights, disable_tqdm=True):
        W_1, b_1, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W_2, b_2 = weights
        h = np.copy(hprev)
        C = np.copy(Cprev)
        h = h.reshape((self.batch_size, self.h_size, 1))
        C = C.reshape((self.batch_size, self.h_size, 1))
        loss = 0
        # W_sth_dropout = get_dropout_function((self.h_size, self.h_size + self.x_size), self.keep_prob)
        # b_sth_dropout = get_dropout_function((self.h_size,), self.keep_prob)
        # W_dropout = get_dropout_function((self.y_size, self.h_size), self.keep_prob)
        # b_dropout = get_dropout_function((self.y_size,), self.keep_prob)
        cell_dropout = get_dropout_function((self.batch_size, self.h_size, 1), self.keep_prob)
        y_dropout = get_dropout_function((self.batch_size, self.y_size, 1), self.keep_prob)
        for t in tqdm(range(len(inputs)), disable=disable_tqdm):
            x = np.array([char_to_one_hot(c) for c in inputs[:, t]])
            x = x.reshape((self.batch_size, -1, 1))

            x = np.matmul(W_1, x) + np.reshape(b_1, (-1, 1))
            x = cell_dropout(x)

            f = sigmoid(np.matmul(W_f, np.concatenate((h, x), axis=1)) + np.reshape(b_f, (-1, 1)))
            f = cell_dropout(f)
            i = sigmoid(np.matmul(W_i, np.concatenate((h, x), axis=1)) + np.reshape(b_i, (-1, 1)))
            i = cell_dropout(i)
            C_hat = np.tanh(np.matmul(W_c, np.concatenate((h, x), axis=1)) + np.reshape(b_c, (-1, 1)))
            C_hat = cell_dropout(C_hat)
            C = f*C + i*C_hat
            C = cell_dropout(C)
            o = sigmoid(np.matmul(W_o, np.concatenate((h, x), axis=1)) + np.reshape(b_o, (-1, 1)))
            o = cell_dropout(o)
            h = o*np.tanh(C)
            h = cell_dropout(h)
            ys = np.matmul(W_2, h) + np.reshape(b_2, (-1, 1))
            ys = y_dropout(ys)

            target_indices = np.array([char_to_index[c] for c in targets[:, t]])
            # ps_target[t] = np.exp(ys[t][target_index])/np.sum(np.exp(ys[t]))  # probability for next chars being target
            # loss += -np.log(ps_target[t])
            loss += np.sum([-(y[target_index] - logsumexp(y)) for y, target_index in zip(ys, target_indices)])/(self.number_of_steps*self.batch_size)
        return loss

    def _d_cost(self, inputs, targets, hprev, Cprev, weights):
        return grad(self._cost, 4)(inputs, targets, hprev, Cprev, weights)

    def _d_cost_batched(self, inputs, targets, hprev, Cprev, weights):
        return grad(self._cost_batched, 4)(inputs, targets, hprev, Cprev, weights)

    def _random_matrix(self, shape):
        return 2*self.init_scale*np.random.random_sample(shape) - self.init_scale

    def perplexity(self, data):
        return np.exp(self._cost(data[:-1], data[1:], self.h, self.C, self.weights, disable_tqdm=False))

    def _update_hidden_state(self, inputs, weights):
        W_1, b_1, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W_2, b_2 = weights
        for t in range(len(inputs)):
            x = char_to_one_hot(inputs[t])

            x = np.matmul(W_1, x) + b_1

            f = sigmoid(np.matmul(W_f, np.concatenate((self.h, x))) + b_f)
            i = sigmoid(np.matmul(W_i, np.concatenate((self.h, x))) + b_i)
            C_hat = np.tanh(np.matmul(W_c, np.concatenate((self.h, x))) + b_c)
            self.C = f*self.C + i*C_hat
            o = sigmoid(np.matmul(W_o, np.concatenate((self.h, x))) + b_o)
            self.h = o*np.tanh(self.C)
        return self.h

    def _update_hidden_state_batched(self, inputs, h, C, weights):
        W_1, b_1, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W_2, b_2 = weights
        h = h.reshape((self.batch_size, self.h_size, 1))
        C = C.reshape((self.batch_size, self.h_size, 1))
        for t in range(len(inputs)):
            x = np.array([char_to_one_hot(c) for c in inputs[:, t]])
            x = x.reshape((self.batch_size, -1, 1))

            x = np.matmul(W_1, x) + np.reshape(b_1, (-1, 1))

            f = sigmoid(np.matmul(W_f, np.concatenate((h, x), axis=1)) + np.reshape(b_f, (-1, 1)))
            i = sigmoid(np.matmul(W_i, np.concatenate((h, x), axis=1)) + np.reshape(b_i, (-1, 1)))
            C_hat = np.tanh(np.matmul(W_c, np.concatenate((h, x), axis=1)) + np.reshape(b_c, (-1, 1)))
            C = f*C + i*C_hat
            o = sigmoid(np.matmul(W_o, np.concatenate((h, x), axis=1)) + np.reshape(b_o, (-1, 1)))
            h = o*np.tanh(C)
        return h, C

    def sample(self, seed, number_of_characters_to_generate):
        h = self.h
        C = self.C
        x = char_to_one_hot(seed)
        ixes = []
        for t in range(number_of_characters_to_generate):
            x = np.matmul(self.W_1, x) + self.b_1
            f = sigmoid(np.matmul(self.W_f, np.concatenate((h, x))) + self.b_f)
            i = sigmoid(np.matmul(self.W_i, np.concatenate((h, x))) + self.b_i)
            C_hat = np.tanh(np.matmul(self.W_c, np.concatenate((h, x))) + self.b_c)
            C = f*C + i*C_hat
            o = sigmoid(np.matmul(self.W_o, np.concatenate((h, x))) + self.b_o)
            h = o*np.tanh(C)
            y = np.matmul(self.W_2, h) + self.b_2

            p = np.exp(y)/np.sum(np.exp(y))
            ix = np.random.choice(range(alphabet_size), p=p)
            x = np.zeros(alphabet_size)
            x[ix] = 1
            ixes.append(ix)
        return indices_to_string(ixes)

    def fit(self):
        training_data_2 = np.array(list(training_data))
        l = len(training_data_2)
        l -= l % self.batch_size

        training_data_2 = training_data_2[:l]
        training_data_2 = np.array(list(training_data_2))
        training_data_2 = training_data_2.reshape((self.batch_size, -1))

        # sanity check, na poczatku powinno byc ~wielkosci alfabetu
        print('validation perplexity:')
        print(self.perplexity(validation_data))
        best_validation_perplexity = 100
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            self.C = np.zeros(self.h_size)
            self.h = np.zeros(self.h_size)
            hprev = np.tile(self.h, (self.batch_size, 1))
            Cprev = np.tile(self.C, (self.batch_size, 1))
            for i in tqdm(range((training_data_2.shape[1]-1)//self.number_of_steps)):
                inputs = training_data_2[:, i*self.number_of_steps:(i+1)*self.number_of_steps]
                targets = training_data_2[:, i*self.number_of_steps+1:(i+1)*self.number_of_steps+1]

                delta_w = self._d_cost_batched(inputs, targets, hprev, Cprev, self.weights)
                clipped_delta_w = [np.clip(d, -5, 5) for d in delta_w]
                # print(any([cdw != dw for cdw, dw in zip(clipped_delta_w, delta_w)]))  doesn't work!
                delta_w = clipped_delta_w
                for w, d in zip(self.weights, delta_w):
                    w -= d*self.learning_rate

                hprev, Cprev = self._update_hidden_state_batched(inputs, hprev, Cprev, self.weights)

            print('validation perplexity:')  # czy powinnam zerowac state? jest po 10 ksiegach
            validation_perplexity = self.perplexity(validation_data)
            print(validation_perplexity)

            if validation_perplexity < best_validation_perplexity:
                best_validation_perplexity = validation_perplexity
            else:
                self.learning_rate /= 2

            prefix = 'Jam jest Jacek'
            self._update_hidden_state(prefix[:-1],
                                      self.weights)  # najpierw wprowadzam prefix ignorujac outputy, nie zaczynam wczytywac ich zaraz po J
            sample = self.sample(prefix[-1], 200)
            print(prefix + sample)

        print('test perplexity:')  # czy powinnam zerowac state? jest po 11 ksiegach i 'Jam jest Jacek'...
        print(self.perplexity(test_data))


params = {'num_of_epochs': 35,
          'learning_rate': 4,
          'init_scale': 0.1,
          'number_of_steps': 20,
          'batch_size': 20,
          'keep_prob': 0.9}
lstm = LSTM(alphabet_size, alphabet_size, **params)
lstm.fit()
