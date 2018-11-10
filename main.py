import numpy as np
from mlp import MLP
import os
import matplotlib.pyplot as plt


def gradient_check(parameter, mlp, perturbation, x, y):
    mlp.forward_pass(x)
    mlp.calculate_cost(y)
    cost = mlp.fprop['cost']
    mlp.backward_pass(x, y)
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


def plot_decision_boundary(clf, x, y, title, save_name):
    fig, ax = plt.subplots()
    x_min, x_max = x[0, :].min() - 0.25, x[0, :].max() + 0.25
    y_min, y_max = x[1, :].min() - 0.25, x[1, :].max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.transpose(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.bwr)
    unique_classes = np.unique(y)
    colors = plt.cm.spring(np.linspace(0, 1, len(unique_classes)))
    for i, class_ in enumerate(unique_classes):
        _, mask = np.where(y == class_)
        ax.scatter(x[0, mask], x[1, mask], c=colors[i], label='Category: {}'.format(class_),
                   s=3, edgecolors='black', linewidths=0.2)
    ax.legend(loc='upper right', framealpha=1)
    ax.grid(True)
    plt.title(title)
    plt.axis("tight")
    plt.savefig(save_name)
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
    with open(os.path.join('results', 'experiment1', 'experiment1.txt'), 'w') as file:
        file.write(str(experiment_1))


def experiment_2():
    inputs = 2
    outputs = 3
    hidden = 3
    n = 2
    lambdas = [0.19, 0.45, 0.12, 0.3]
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
    with open(os.path.join('results', 'experiment2', 'experiment2.txt'), 'w') as file:
        file.write(str(experiment_2))


def experiment_5():
    # constants
    save_dir = os.path.join('results', 'experiment5')

    # Load data
    circle = np.loadtxt(os.path.join('data', 'circle.txt'))
    x = circle[:, :-1].T
    y = np.clip(circle[:, -1].T.reshape(1, -1), 0, 1).astype(int)

    # Varying weight decay
    lr = 0.001
    lambdas = [([0.0, 0.0, 0.0, 0.0], 'Decision surface on an MLP for circles with no regularization'),
               ([0.002, 0.0002, 0.002, 0.002], 'Decision surface on an MLP for circles with low regularization (L1 & L2)'),
               ([0.005, 0.005, 0.005, 0.005], 'Decision surface on an MLP for circles with high regularization (L1 & L2)'),
               ([0.01, 0.0, 0.01, 0.0], 'Decision surface on an MLP for circles with low L1 regularization'),
               ([0.015, 0.0, 0.015, 0.0], 'Decision surface on an MLP for circles with high L1 regularization'),
               ([0.0, 0.015, 0.0, 0.015], 'Decision surface on an MLP for circles with low L2 regularization'),
               ([0.0, 0.02, 0.0, 0.02], 'Decision surface on an MLP for circles with low L2 regularization')]
    hidden = 200
    epochs = 50
    batch_size = 500
    for i, (lambda_, title) in enumerate(lambdas):
        inputs = len(x)
        outputs = len(np.unique(y))
        mlp = MLP(inputs, hidden, outputs, lambda_)
        mlp.fit(x, y, batch_size, epochs, lr)
        name = os.path.join(save_dir, 'lambda', 'lambda_{}'.format(i))
        plot_decision_boundary(mlp, x, y, title, name)

    # Varying number of hidden units
    lr = 0.001
    lambdas = [0.001, 0.0001, 0.001, 0.001]
    hiddens = [1, 2, 5, 20, 50, 100, 200]
    epochs = 50
    batch_size = 500
    for i, hidden in enumerate(hiddens):
        inputs = len(x)
        outputs = len(np.unique(y))
        mlp = MLP(inputs, hidden, outputs, lambdas)
        mlp.fit(x, y, batch_size, epochs, lr)
        name = os.path.join(save_dir, 'hidden', 'hidden_{}'.format(i))
        title = 'Decision surface on an MLP for circles with {} hidden unit(s)'.format(hidden)
        plot_decision_boundary(mlp, x, y, title, name)

    # Varying number of epochs
    lr = 0.001
    lambdas = [0.001, 0.0001, 0.001, 0.001]
    hidden = 100
    epochs = [1, 2, 3, 10, 20, 50, 100]
    batch_size = 500
    for i, epoch in enumerate(epochs):
        inputs = len(x)
        outputs = len(np.unique(y))
        mlp = MLP(inputs, hidden, outputs, lambdas)
        mlp.fit(x, y, batch_size, epoch, lr)
        name = os.path.join(save_dir, 'epochs', 'epochs_{}'.format(i))
        title = 'Decision surface on an MLP for circles with {} epoch(s)'.format(epoch)
        plot_decision_boundary(mlp, x, y, title, name)


def experiment_10():
    pass


if __name__ == '__main__':
    # EXPERIMENTS

    experiment_5()


