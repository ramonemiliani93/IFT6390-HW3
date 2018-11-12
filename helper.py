import numpy as np


def initialize_params(n_inputs, n_hidden, n_outputs):
    w1 = np.random.uniform(low=-1 / np.sqrt(n_inputs), high=1 / np.sqrt(n_inputs), size=(n_hidden, n_inputs))
    w2 = np.random.uniform(low=-1 / np.sqrt(n_hidden), high=1 / np.sqrt(n_hidden), size=(n_outputs, n_hidden))
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


def initialize_gradient_results(n_inputs, n_hidden, n_outputs):
    w1 = np.zeros((n_hidden, n_inputs))
    w2 = np.zeros((n_outputs, n_hidden))
    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_outputs, 1))
    dict_ = {'grad_w1': w1, 'grad_w2': w2, 'grad_b1': b1, 'grad_b2': b2}
    return dict_


def save_log(filename, losses, errors, dh, lr, K):
    '''
    Save the losses and errors to a txt file specified
    by filename.
    '''

    if 'test' in losses and 'val' in losses:
        losses_cat = np.c_[np.arange(len(losses['train']), dtype='int') + 1,
                           losses['train'], losses['val'], losses['test']]
        errors_cat = np.c_[errors['train'], errors['val'], errors['test']]
        full_cat = np.c_[losses_cat, errors_cat]
        header1 = 'dh=' + str(dh) + ', lr=' + str(
            lr) + ', K=' + str(K)
        header2 = 'Epoch, train loss, val loss, test loss, train error, val error, test error'
        header = header1 + '\n' + header2
    elif 'val' in losses:
        losses_cat = np.c_[np.arange(len(losses['train']), dtype='int') + 1,
                           losses['train'], losses['val']]
        errors_cat = np.c_[errors['train'], errors['val']]
        full_cat = np.c_[losses_cat, errors_cat]
        header1 = 'dh=' + str(dh) + ', lr=' + str(
            lr) + ', K=' + str(K)
        header2 = 'Epoch, train loss, val loss, train error, val error'
        header = header1 + '\n' + header2
    else:
        losses_cat = np.c_[np.arange(len(losses['train']), dtype='int') + 1,
                           losses['train']]
        errors_cat = np.c_[errors['train']]
        full_cat = np.c_[losses_cat, errors_cat]
        header1 = 'dh=' + str(dh) + ', lr=' + str(
            lr) + ', K=' + str(K)
        header2 = 'Epoch, train loss, train error'
        header = header1 + '\n' + header2
    np.savetxt(filename, full_cat, fmt='%3.6f', header=header)
