import autograd.numpy as np

from autograd import grad
from autograd.scipy.misc import logsumexp
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
alphabet = sorted(set(training_data + validation_data + test_data))   # czy powinnam tutaj uzywac danych walidacyjnych i testowych,
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


# print(alphabet)
# print(char_to_index)
# print(char_to_one_hot('J'))
# print(char_to_one_hot('J').shape)
# print(len(string_to_one_hots('Jam jest Jacek')))


class RecurrentNeuralNetwork:

    def __init__(self, x_size, y_size, num_of_epochs, learning_rate, init_scale, number_of_steps):
        self.num_of_epochs = num_of_epochs
        self.hidden_size = 200
        batch_size = 20
        self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        self.init_scale = init_scale
        self.h = np.zeros(self.hidden_size)
        self.W_hh = self._random_matrix((self.hidden_size, self.hidden_size))
        self.W_xh = self._random_matrix((self.hidden_size, x_size))
        self.W_hy = self._random_matrix((y_size, self.hidden_size))
        self.weights = [self.W_hh, self.W_xh, self.W_hy]

    def _random_matrix(self, shape):
        return 2*self.init_scale*np.random.random_sample(shape) - self.init_scale

    def step(self, x):
        self.h = np.tanh(np.matmul(self.W_hh, self.h) + np.matmul(self.W_xh, x))
        return np.matmul(self.W_hy, self.h)

    def _d_cost(self, inputs, targets, hprev, weights):
        return grad(self._cost, 3)(inputs, targets, hprev, weights)

    def _cost(self, inputs, targets, hprev, weights, disable_tqdm=True):
        W_hh, W_xh, W_hy = weights
        h = np.copy(hprev)
        loss = 0
        for t in tqdm(range(len(inputs)), disable=disable_tqdm):
            x = char_to_one_hot(inputs[t])
            h = np.tanh(np.matmul(W_hh, h) + np.matmul(W_xh, x))
            y = np.matmul(W_hy, h)
            target_index = char_to_index[targets[t]]
            # ps_target[t] = np.exp(ys[t][target_index])/np.sum(np.exp(ys[t]))  # probability for next chars being target
            # loss += -np.log(ps_target[t])
            loss += -(y[target_index] - logsumexp(y))

        loss = loss/len(inputs)
        return loss

    def _get_new_hidden_state(self, inputs, hprev, weights):
        W_hh, W_xh, W_hy = weights
        h = np.copy(hprev)
        for t in range(len(inputs)):
            x = char_to_one_hot(inputs[t])
            h = np.tanh(np.matmul(W_hh, h) + np.matmul(W_xh, x))
            y = np.matmul(W_hy, h)
        return h

    def sample(self, seed, number_of_characters_to_generate):
        h = self.h
        x = char_to_one_hot(seed)
        ixes = []
        for t in range(number_of_characters_to_generate):
            h = np.tanh(np.matmul(self.W_hh, h) + np.matmul(self.W_xh, x))
            y = np.matmul(self.W_hy, h)

            p = np.exp(y)/np.sum(np.exp(y))
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
            self.h = np.zeros(self.hidden_size)
            for i in tqdm(range((len(training_data)-1)//self.number_of_steps)):
                inputs = training_data[i*self.number_of_steps:(i+1)*self.number_of_steps]
                targets = training_data[i*self.number_of_steps+1:(i+1)*self.number_of_steps+1]

                delta_w = self._d_cost(inputs, targets, self.h, self.weights)
                clipped_delta_w = [np.clip(d, -5, 5) for d in delta_w]
                # print(any([cdw != dw for cdw, dw in zip(clipped_delta_w, delta_w)]))  doesn't work!
                delta_w = clipped_delta_w
                for w, d in zip(self.weights, delta_w):
                    w -= d*self.learning_rate

                self.h = self._get_new_hidden_state(inputs, self.h, self.weights)

            print('validation perplexity:')  # czy powinnam zerowac state? jest po 10 ksiegach
            print(self.perplexity(validation_data))

            prefix = 'Jam jest Jacek'
            self.h = self._get_new_hidden_state(prefix[:-1], self.h, self.weights)  # najpierw wprowadzam prefix ignorujac outputy, nie zaczynam wczytywac ich zaraz po J
            sample = self.sample(prefix[-1], 200)
            print(prefix + sample)

        print('test perplexity:')  # czy powinnam zerowac state? jest po 11 ksiegach i 'Jam jest Jacek'...
        print(self.perplexity(test_data))

    def perplexity(self, data):
        return np.exp(self._cost(data[:-1], data[1:], self.h, self.weights, disable_tqdm=False))


params = {'num_of_epochs': 10,
          'learning_rate': 0.05,
          'init_scale': 0.1,
          'number_of_steps': 25}
rnn = RecurrentNeuralNetwork(alphabet_size, alphabet_size, **params)
rnn.fit()



# (i+1)*self.number_of_steps+1 <= len
# (i+1)*self.number_of_steps <= len-1
# (i+1) <= (len-1)/self.number_of_steps
# i <= (len-1)/self.number_of_steps - 1
# i < (len-1)/self.number_of_steps



# TODO

# batch_size

# proszę sprawdzić jak wiele pomagają tzw. embeddings; chodzi tu o to aby zamiast one-hot-wektora literki na wejściu podawać jej reprezentację tej literki jako wektor powiedzmy długości hidden_size;
# efektywnie przemnażamy one-hot-wektor przez macierz wielkości vocab_size x hidden_size - ???
# embedding na literkach - przemnożenie przez macierz i użycie fcji aktywacji - np. dropout
# embedding odwrotny - nie liczyc odwrotnej, tylko stransponowac te macierz i pomnozyc output przez nia - czy tutaj jest jakaś fcja aktywacji?
# w pracy regularyzacja l2 zamiast dropoutu

# gradient clipping: spróbuj obcinać gradient do jakiejś stałej, np. 6? sprawdź czy to pomaga i jak często clipping jest odpalany

# wizualizacje