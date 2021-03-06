import numpy as np

from extract_data import extract_images, extract_labels
from simple_nn_tf import SimpleNN
from utils import shuffle, train_validation_split, compute_accuracy

if __name__ == "__main__":
    images = extract_images('train-images-idx3-ubyte.gz')
    images = np.reshape(images, (-1, 28*28))/255
    labels = extract_labels('train-labels-idx1-ubyte.gz', one_hot=True)

    images, labels = shuffle(images, labels)

    training_set_size = 55_000
    training_images, training_labels, valid_images, valid_labels \
        = train_validation_split(images, labels, training_set_size)

    params = {'batch_size': 64,
              'num_of_epochs': 40,
              'learning_rate': 0.1,
              'init_scale': 0.05,
              'keep_prob': 0.75,
              'ema': 0.999}
    model = SimpleNN([28*28, 500, 10], **params)
    model.fit(training_images, training_labels, valid_images, valid_labels)

    eval_images = extract_images('t10k-images-idx3-ubyte.gz')
    eval_images = np.reshape(eval_images, (-1, 28*28))/255
    eval_labels = extract_labels('t10k-labels-idx1-ubyte.gz')

    predictions = model.predict(eval_images)

    print("test accuracy: " + str(round(compute_accuracy(predictions, eval_labels), 2)))

