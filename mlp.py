import numpy as np
from sklearn import datasets
from helper import initialize_params
from helper import initialize_activations
from helper import initialize_variables
from helper import initialize_gradient_results
from activation import Relu
from activation import Softmax
from cost import LogCost


class MLP:

    def __init__(self, n_inputs, n_hidden, n_outputs, lambdas, matrix=True):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.lambdas = lambdas
        self.matrix = matrix
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
        # The penalization has to be added the same number of times as the number of examples!
        reps = grad_oa.shape[-1]
        grad_w2 = np.matmul(grad_oa, np.transpose(self.fprop['hs']))
        grad_w2_pen = reps*(self.lambdas[2]*np.sign(self.params['w2']) + self.lambdas[3]*2*(self.params['w2']))
        self.bprop['grad_w2'] = grad_w2 + grad_w2_pen


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
        # The penalization has to be added the same number of times as the number of examples!
        grad_w1 = np.matmul(grad_ha, np.transpose(x))
        grad_w1_pen = reps*(self.lambdas[0]*np.sign(self.params['w1']) + self.lambdas[1]*2*(self.params['w1']))
        self.bprop['grad_w1'] = grad_w1 + grad_w1_pen

    def calculate_gradient(self, x, y):
        gradients = initialize_gradient_results(self.n_inputs, self.n_hidden, self.n_outputs)
        if self.matrix:
            self.forward_pass(x)
            self.calculate_cost(y)
            cost = self.fprop['cost']
            self.backward_pass(x, y)
            for key in gradients.keys():
                gradients[key] = self.bprop[key]
        else:
            cost = []
            for i in range(x.shape[-1]):
                self.forward_pass(x[:, [i]])
                self.calculate_cost(y[:, [i]])
                cost.append(np.asscalar(self.fprop['cost']))
                self.backward_pass(x[:, [i]], y[:, [i]])
                for key in gradients.keys():
                    gradients[key] += self.bprop[key]
            cost = np.asarray(cost)
        return cost, gradients

    def update_weights(self, eta, gradients):
        for key in self.params.keys():
            self.params[key] = self.params[key] - eta*gradients['grad_{}'.format(key)]

    def fit(self, x_train, y_train, K, epochs, eta, x_val=None, y_val=None, x_test=None, y_test=None):
        assert x_train.shape[-1] == y_train.shape[-1]
        train_cost = []
        val_cost = []
        test_cost = []
        train_error = []
        val_error = []
        test_error = []
        n_batches = x_train.shape[-1]/K
        for j in range(epochs):
            cost_batch = []
            for i in range(int(np.floor(n_batches))):
                x_batch = x_train[:, i*K:(i+1)*K]
                y_batch = y_train[:, i*K:(i+1)*K]
                cost, gradients = self.calculate_gradient(x_batch, y_batch)
                cost_batch.append(np.sum(cost))
                self.update_weights(eta, gradients)
            if not n_batches.is_integer():
                x_batch = x_train[:, int(n_batches*K):]
                y_batch = y_train[:, int(n_batches*K):]
                cost, gradients = self.calculate_gradient(x_batch, y_batch)
                self.update_weights(eta, gradients)
                cost_batch.append(np.sum(cost))
            train_cost.append(np.sum(cost_batch)/x_train.shape[-1])

            y_pred = self.predict(x_train)
            train_error.append(self.classification_error(y_pred, y_train))

            if x_val is not None and x_test is not None:
                self.forward_pass(x_val)
                self.calculate_cost(y_val)
                cost_val = self.fprop['cost']
                val_cost.append(np.sum(cost_val) / x_val.shape[-1])

                self.forward_pass(x_test)
                self.calculate_cost(y_test)
                cost_test = self.fprop['cost']
                test_cost.append(np.sum(cost_test)/x_test.shape[-1])

                y_pred_val = self.predict(x_val)
                val_error.append(self.classification_error(y_pred_val, y_val))
                y_pred_test = self.predict(x_test)
                test_error.append(self.classification_error(y_pred_test, y_test))

        return train_cost, train_error, val_cost, val_error, test_cost, test_error

    def predict(self, x):
        prediction = self.forward_pass(x)
        prediction = np.argmax(prediction, axis=0)
        return prediction

    def classification_error(self, y_pred, y):
        y_pred = np.expand_dims(y_pred, axis=-1).T
        correct = y_pred == y
        error = 1 - (np.sum(correct)/y.shape[1])
        return error


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
