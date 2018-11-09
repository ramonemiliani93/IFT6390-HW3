import numpy as np


class Activation(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Relu(Activation):
    def forward(self, x):
        relu = np.clip(x, a_min=0, a_max=None)
        return relu


class Softmax(Activation):
    def forward(self, x):
        max = np.amax(x)
        softmax = np.exp(x-max)/np.sum(np.exp(x-max), axis=0)
        return softmax
