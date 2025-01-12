import pickle
import gzip
import os

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "../data/mnist.pkl.gz")

# load 50,000 dataset from mnist
def load_data():
    # training data is 28*28 pixel
    # return value =: training data, the validation data, and the test data.

    print("loading dataset...")
    f = gzip.open(path, "rb")

    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    training_data, validation_data, test_data = u.load()
    f.close()
    print("loaded")
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    print("reshapping...")
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    print("completed")
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    This is used to convert a digit
    (0...9) into a corresponding desired output
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
