import numpy as np


def initialize_params(n_inputs, n_hidden, n_outputs):
    w1 = np.random.uniform(low=-1/np.sqrt(n_inputs), high=1/np.sqrt(n_inputs), size=(n_hidden, n_inputs))
    w2 = np.random.uniform(low=-1/np.sqrt(n_hidden), high=1/np.sqrt(n_hidden), size=(n_outputs, n_hidden))
    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_outputs, 1))
    dict_ = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
    return dict_


def initialize_activations(class_a1, class_a2):
    a1 = class_a1()
    a2 = class_a2()
    dict_ = {'a1': a1, 'a2': a2}
    return dict_


def initialize_variables(names, initialization=None):
    dict_ = {}
    for i in names:
        dict_[i] = initialization
    return dict_