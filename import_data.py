import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


def import_endgame(filename):
    npzfile2 = np.load(filename)
    #    print("npz variables")
    #    print(npzfile.files)
    #    print(npzfile['y_train'])
    X_train = npzfile2['X_train']
    y_train = npzfile2['y_train']

    # Converting time to mate back to an number (0,x)
    y_train[y_train[:, 3] < 0, 3] = (-2000 - y_train[y_train[:, 3] < 0, 3]) / 2
    y_train[y_train[:, 3] > 0, 3] = (2001 - y_train[y_train[:, 3] > 0, 3]) / 2
    y_train[:, 3] += 16

    # Or, convert the int values into floats, this doesn't seem to matter
    # X_train = X_train.astype(float)
    # y_train = y_train.astype(float)

    print("frequency list:")
    unique_elements, counts_elements = np.unique(y_train[:, 3], return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))
    return X_train, y_train
