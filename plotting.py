import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrices(y_test, y_test_pred,
                            classes, normalize=False):

    plt.subplots(1, 2, num='Confusion matrices', figsize=(12, 6))

    cnf_matrix = confusion_matrix(y_test, y_test_pred)

    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cnf_matrix, classes,
                          title='Confusion matrix, training set',
                          normalize=normalize)

    plt.subplot(1, 2, 2)
    np.fill_diagonal(cnf_matrix, 0)
    plot_confusion_matrix(cnf_matrix, classes,
                          title='Confusion matrix w/o diagonal, testing set',
                          normalize=normalize)

    plt.show()
