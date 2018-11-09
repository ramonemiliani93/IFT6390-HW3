import numpy as np
from mlp import MLP
import os
import matplotlib.pyplot as plt


def gradient_check(parameter, mlp, perturbation, x, y):
    mlp.forward_pass(x)
    mlp.calculate_cost(y)
    mlp.backward_pass(x, y)
    cost = mlp.fprop['cost']
    numerical_gradient = np.zeros((mlp.params[parameter].shape[0], mlp.params[parameter].shape[1], len(cost)))
    rows, columns = mlp.params[parameter].shape
    for i in range(rows):
        for j in range(columns):
            mlp.params[parameter][i, j] = mlp.params[parameter][i, j] + perturbation
            mlp.forward_pass(x)
            mlp.calculate_cost(y)
            numerical_gradient[i, j, :] = mlp.fprop['cost']-cost
            mlp.params[parameter][i, j] = mlp.params[parameter][i, j] - perturbation
    numerical_gradient = numerical_gradient/perturbation
    numerical_gradient = np.sum(numerical_gradient, axis=2)
    check = np.divide(numerical_gradient+perturbation*0.001, mlp.bprop['grad_{}'.format(parameter)]+perturbation*0.001)
    return numerical_gradient, check

def plot_decision_boundary(clf, x, y):
    x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
    y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.transpose(np.c_[xx.ravel(), yy.ravel()]))
    print(Z.shape)
    print(np.sum(Z))
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.bwr)
    plt.scatter(x[0, :], x[1, :], c=y.ravel(), s=2, cmap=plt.cm.spring)
    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()



def experiment_1():
    inputs = 4
    outputs = 6
    hidden = 3
    n = 1
    lambdas = [0.1, 0.1, 0.1, 0.1]
    mlp = MLP(inputs, hidden, outputs, lambdas)
    x = np.random.rand(inputs, n)
    y = np.random.randint(0, outputs, (1, n))
    numerical_gradient_w2, check_w2 = gradient_check('w2', mlp, 10e-5, x, y)
    numerical_gradient_b2, check_b2 = gradient_check('b2', mlp, 10e-5, x, y)
    numerical_gradient_w1, check_w1 = gradient_check('w1', mlp, 10e-5, x, y)
    numerical_gradient_b1, check_b1 = gradient_check('b1', mlp, 10e-5, x, y)
    experiment_1 = {
        'numerical_gradient_w2': numerical_gradient_w2,
        'grad_w2': mlp.bprop['grad_w2'],
        'check_w2': check_w2,
        'numerical_gradient_b2': numerical_gradient_b2,
        'grad_b2': mlp.bprop['grad_b2'],
        'check_b2': check_b2,
        'numerical_gradient_w1': numerical_gradient_w1,
        'grad_w1': mlp.bprop['grad_w1'],
        'check_w1': check_w1,
        'numerical_gradient_b1': numerical_gradient_b1,
        'grad_b1': mlp.bprop['grad_b1'],
        'check_b1': check_b1}
    np.save(os.path.join('results', 'experiment_1.npy'), experiment_1)


def experiment_2():
    inputs = 2
    outputs = 3
    hidden = 3
    n = 1
    lambdas = [0.1, 0.1, 0.1, 0.1]
    mlp = MLP(inputs, hidden, outputs, lambdas)
    x = np.random.rand(inputs, n)
    y = np.random.randint(0, outputs, (1, n))
    numerical_gradient_w2, check_w2 = gradient_check('w2', mlp, 10e-5, x, y)
    numerical_gradient_b2, check_b2 = gradient_check('b2', mlp, 10e-5, x, y)
    numerical_gradient_w1, check_w1 = gradient_check('w1', mlp, 10e-5, x, y)
    numerical_gradient_b1, check_b1 = gradient_check('b1', mlp, 10e-5, x, y)
    experiment_2 = {
        'numerical_gradient_w2': numerical_gradient_w2,
        'grad_w2': mlp.bprop['grad_w2'],
        'check_w2': check_w2,
        'numerical_gradient_b2': numerical_gradient_b2,
        'grad_b2': mlp.bprop['grad_b2'],
        'check_b2': check_b2,
        'numerical_gradient_w1': numerical_gradient_w1,
        'grad_w1': mlp.bprop['grad_w1'],
        'check_w1': check_w1,
        'numerical_gradient_b1': numerical_gradient_b1,
        'grad_b1': mlp.bprop['grad_b1'],
        'check_b1': check_b1}
    np.save(os.path.join('results', 'experiment_1.npy'), experiment_2)

def experiment_5():
    # Circle data
    circle = np.loadtxt(os.path.join('data', 'circle.txt'))
    x = circle[:, :-1].T
    y = np.clip(circle[:, -1].T.reshape(1, -1), 0, 1).astype(int)
    inputs = len(x)
    outputs = len(np.unique(y))
    hidden = 5000
    mlp = MLP(inputs, hidden, outputs, [0, 0, 0, 0])
    mlp.fit(x, y, 10, 100, 0.001)
    plot_decision_boundary(mlp, x, y)

if __name__ == '__main__':
    # EXPERIMENTS

    experiment_5()

