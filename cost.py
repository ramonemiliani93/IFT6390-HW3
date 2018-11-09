import numpy as np


class Cost(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self, predictions, y, params, lambdas):
        raise NotImplementedError

    def backward(self, predictions, y):
        raise NotImplementedError


class LogCost(Cost):

    def forward(self, predictions, y, params, lambdas):
        n_predictions = predictions.shape[-1]
        onehot = np.zeros((self.n_classes, n_predictions))
        onehot[y, np.arange(n_predictions)] = 1
        cost = np.sum(np.multiply(predictions, onehot), axis=0)
        cost = -np.log(cost) + lambdas[0]*np.sum(np.sum(np.abs(params['w1']))) + \
               lambdas[1]*np.sum(np.sum(np.square(params['w1']))) + \
               lambdas[2]*np.sum(np.sum(np.abs(params['w2']))) + \
               lambdas[3]*np.sum(np.sum(np.square(params['w2'])))
        return cost

    def backward(self, predictions, y):
        n_predictions = predictions.shape[-1]
        onehot = np.zeros((self.n_classes, n_predictions))
        onehot[y, np.arange(n_predictions)] = 1
        cost = predictions - onehot
        return cost


if __name__ == '__main__':
    logcost = LogCost(4)
    print(logcost.forward(np.array([0.9, 0.01, 0.01, 0.08]).reshape(-1, 1), np.array([0])))
    print(logcost.backward(np.array([0.9, 0.01, 0.01, 0.08]).reshape(-1, 1), np.array([0])))
