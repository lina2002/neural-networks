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


class LSTM:

    def __init__(self, x_size, y_size, num_of_epochs, learning_rate, init_scale, number_of_steps):
        self.num_of_epochs = num_of_epochs
        self.h_size = h_size = y_size
        self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        self.init_scale = init_scale
        self.C = np.zeros(h_size)
        self.h = np.zeros(h_size)
        self.W_f = self._random_matrix((h_size, h_size + x_size))
        self.b_f = self._random_matrix((h_size,)) + 1
        self.W_i = self._random_matrix((h_size, h_size + x_size))
        self.b_i = self._random_matrix((h_size,))
        self.W_c = self._random_matrix((h_size, h_size + x_size))
        self.b_c = self._random_matrix((h_size,))
        self.W_o = self._random_matrix((h_size, h_size + x_size))
        self.b_o = self._random_matrix((h_size,))
        self.weights = [self.W_f, self.b_f, self.W_i, self.b_i, self.W_c, self.b_c, self.W_o, self.b_o]

    def _cost(self, inputs, targets, hprev, Cprev, weights, disable_tqdm=True):
        W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o = weights
        h = np.copy(hprev)
        C = np.copy(Cprev)
        loss = 0
        for t in tqdm(range(len(inputs)), disable=disable_tqdm):
            x = char_to_one_hot(inputs[t])

            f = sigmoid(np.matmul(W_f, np.concatenate((h, x))) + b_f)
            i = sigmoid(np.matmul(W_i, np.concatenate((h, x))) + b_i)
            C_hat = np.tanh(np.matmul(W_c, np.concatenate((h, x))) + b_c)
            C = f*C + i*C_hat
            o = sigmoid(np.matmul(W_o, np.concatenate((h, x))) + b_o)
            h = o*np.tanh(C)

            target_index = char_to_index[targets[t]]
            # ps_target[t] = np.exp(ys[t][target_index])/np.sum(np.exp(ys[t]))  # probability for next chars being target
            # loss += -np.log(ps_target[t])
            loss += -(h[target_index] - logsumexp(h))

        loss = loss/len(inputs)
        return loss

    def _d_cost(self, inputs, targets, hprev, Cprev, weights):
        return grad(self._cost, 4)(inputs, targets, hprev, Cprev, weights)

    def _random_matrix(self, shape):
        return 2*self.init_scale*np.random.random_sample(shape) - self.init_scale

    def perplexity(self, data):
        return np.exp(self._cost(data[:-1], data[1:], self.h, self.C, self.weights, disable_tqdm=False))

    def _get_new_hidden_state(self, inputs, hprev, Cprev, weights):
        W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o = weights
        h = np.copy(hprev)
        C = np.copy(Cprev)
        for t in range(len(inputs)):
            x = char_to_one_hot(inputs[t])

            f = sigmoid(np.matmul(W_f, np.concatenate((h, x))) + b_f)
            i = sigmoid(np.matmul(W_i, np.concatenate((h, x))) + b_i)
            C_hat = np.tanh(np.matmul(W_c, np.concatenate((h, x))) + b_c)
            C = f*C + i*C_hat
            o = sigmoid(np.matmul(W_o, np.concatenate((h, x))) + b_o)
            h = o*np.tanh(C)
        return h

    def sample(self, seed, number_of_characters_to_generate):
        h = self.h
        C = self.C
        x = char_to_one_hot(seed)
        ixes = []
        for t in range(number_of_characters_to_generate):
            f = sigmoid(np.matmul(self.W_f, np.concatenate((h, x))) + self.b_f)
            i = sigmoid(np.matmul(self.W_i, np.concatenate((h, x))) + self.b_i)
            C_hat = np.tanh(np.matmul(self.W_c, np.concatenate((h, x))) + self.b_c)
            C = f*C + i*C_hat
            o = sigmoid(np.matmul(self.W_o, np.concatenate((h, x))) + self.b_o)
            h = o*np.tanh(C)

            p = np.exp(h)/np.sum(np.exp(h))
            ix = np.random.choice(range(alphabet_size), p=p)
            x = np.zeros(alphabet_size)
            x[ix] = 1
            ixes.append(ix)
        return indices_to_string(ixes)

    def fit(self):
        # sanity check, na poczatku powinno byc ~wielkosci alfabetu
        print('validation perplexity:')
        print(self.perplexity(validation_data))
        for epoch in range(self.num_of_epochs):
            print("epoch number: " + str(epoch + 1))
            self.C = np.zeros(self.h_size)
            self.h = np.zeros(self.h_size)
            for i in tqdm(range((len(training_data) - 1)//self.number_of_steps)):
                inputs = training_data[i*self.number_of_steps:(i + 1)*self.number_of_steps]
                targets = training_data[i*self.number_of_steps + 1:(i + 1)*self.number_of_steps + 1]

                delta_w = self._d_cost(inputs, targets, self.h, self.C, self.weights)
                clipped_delta_w = [np.clip(d, -5, 5) for d in delta_w]
                # print(any([cdw != dw for cdw, dw in zip(clipped_delta_w, delta_w)]))  doesn't work!
                delta_w = clipped_delta_w
                for w, d in zip(self.weights, delta_w):
                    w -= d*self.learning_rate

                self.h = self._get_new_hidden_state(inputs, self.h, self.C, self.weights)

            print('validation perplexity:')  # czy powinnam zerowac state? jest po 10 ksiegach
            print(self.perplexity(validation_data))

            prefix = 'Jam jest Jacek'
            self.h = self._get_new_hidden_state(prefix[:-1], self.h, self.C,
                                                self.weights)  # najpierw wprowadzam prefix ignorujac outputy, nie zaczynam wczytywac ich zaraz po J
            sample = self.sample(prefix[-1], 200)
            print(prefix + sample)

        print('test perplexity:')  # czy powinnam zerowac state? jest po 11 ksiegach i 'Jam jest Jacek'...
        print(self.perplexity(test_data))


params = {'num_of_epochs': 10,
          'learning_rate': 0.05,
          'init_scale': 0.1,
          'number_of_steps': 25}
lstm = LSTM(alphabet_size, alphabet_size, **params)
lstm.fit()
