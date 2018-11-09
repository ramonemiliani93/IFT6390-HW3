import numpy as np
from sklearn import datasets
from helper import initialize_params
from helper import initialize_activations
from helper import initialize_variables
from activation import Relu
from activation import Softmax
from cost import LogCost


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_outputs, lambdas):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.lambdas = lambdas
        self.params = initialize_params(n_inputs, n_hidden, n_outputs)
        self.activations = initialize_activations(Relu, Softmax)
        self.cost = LogCost(self.n_outputs)
        self.fprop = initialize_variables(['ha', 'hs', 'oa', 'os', 'cost', 'total_cost'])
        self.bprop = initialize_variables(['grad_oa', 'grad_b2', 'grad_w2', 'grad_hs', 'grad_ha', 'grad_b1', 'grad_w1'])

    def forward_pass(self, x):
        # Hidden layer Wx+b
        ha = np.add(np.matmul(self.params['w1'], x), self.params['b1'])
        # Save result ha
        self.fprop['ha'] = ha

        # Hidden layer activation
        hs = self.activations['a1'].forward(ha)
        # Save result hs
        self.fprop['hs'] = hs

        # Output layer Wx+b
        oa = np.add(np.matmul(self.params['w2'], hs), self.params['b2'])
        # Save result oa
        self.fprop['oa'] = oa

        # Output layer activation
        os = self.activations['a2'].forward(oa)
        # Save result os
        self.fprop['os'] = os

        return os

    def calculate_cost(self, y):
        # Calculate cost
        cost = self.cost.forward(self.fprop['os'], y, self.params, self.lambdas)
        self.fprop['cost'] = cost
        self.fprop['total_cost'] = np.sum(cost)

        return cost

    def backward_pass(self, x, y):
        # Gradient of loss wrt output activation
        grad_oa = self.cost.backward(self.fprop['os'], y)
        self.bprop['grad_oa'] = grad_oa

        # Gradient of loss wrt bias 2
        # Add np.sum over axis 1 when batch
        grad_b2 = np.sum(grad_oa, axis=1, keepdims=True)
        self.bprop['grad_b2'] = grad_b2

        # Gradient of loss wrt weights 2
        grad_w2 = np.matmul(grad_oa, np.transpose(self.fprop['hs']))
        self.bprop['grad_w2'] = grad_w2 + \
                                self.lambdas[2]*np.sign(self.params['w2']) + self.lambdas[3]*2*(self.params['w2'])

        # Gradient of loss wrt hs
        grad_hs = np.matmul(np.transpose(self.params['w2']), grad_oa)
        self.bprop['grad_hs'] = grad_hs

        # Gradient of loss wrt ha
        clip_ha = np.clip(self.fprop['ha'], 0, None)
        clip_ha[clip_ha > 0] = 1
        grad_ha = np.multiply(grad_hs, clip_ha)
        self.bprop['grad_ha'] = grad_ha

        # Gradient of loss wrt bias 1
        # Add np.sum over axis 1 when batch
        grad_b1 = np.sum(grad_ha, axis=1, keepdims=True)
        self.bprop['grad_b1'] = grad_b1

        # Gradient of loss wrt weights 1
        grad_w1 = np.matmul(grad_ha, np.transpose(x))
        self.bprop['grad_w1'] = grad_w1 + \
                                self.lambdas[0]*np.sign(self.params['w1']) + self.lambdas[1]*2*(self.params['w1'])

    def update_weights(self, eta):
        for key in self.params.keys():
            self.params[key] = self.params[key] - eta*self.bprop['grad_{}'.format(key)]

    def fit(self, x, y, K, epochs, eta):
        assert x.shape[-1] == y.shape[-1]
        cost = []
        n_batches = x.shape[-1]/K
        for j in range(epochs):
            for i in range(int(np.floor(n_batches))):
                x_batch = x[:, i*K:(i+1)*K]
                y_batch = y[:, i*K:(i+1)*K]
                self.forward_pass(x_batch)
                self.calculate_cost(y_batch)
                cost.append(self.fprop['total_cost'])
                self.backward_pass(x_batch, y_batch)
                self.update_weights(eta)
            if not n_batches.is_integer():
                x_batch = x[:, int(n_batches*K):]
                y_batch = y[:, int(n_batches*K):]
                self.forward_pass(x_batch)
                self.calculate_cost(y_batch)
                cost.append(self.fprop['total_cost'])
                self.backward_pass(x_batch, y_batch)
                self.update_weights(eta)
            cost.append(self.fprop['total_cost'])
        return cost

    def predict(self, x):
        prediction = self.forward_pass(x)
        print(prediction)
        prediction = np.argmax(prediction, axis=0)
        return prediction


if __name__ == '__main__':
    iris = datasets.load_iris()
    x = np.transpose(iris.data)
    y = iris.target.reshape(1, -1)
    inputs = len(x)
    outputs = len(np.unique(y))
    hidden = 4
    mlp = MLP(inputs, hidden, outputs, [0.1, 0.3, 0.1, 0.3])
    cost = mlp.fit(x, y, 10, 10, 0.2)
    print(cost)
