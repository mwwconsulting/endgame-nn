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
    return (X_train, y_train)
