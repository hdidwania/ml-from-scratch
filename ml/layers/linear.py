"""
Implementation of linear layer
"""

import numpy as np

from ml.layers.base import LearnableLayer


class Linear(LearnableLayer):
    def __init__(self, in_dim, out_dim, init_fn=None):
        super().__init__()
        if not init_fn:
            self.params["w"] = np.random.randn(in_dim, out_dim)
        else:
            self.params["w"] = init_fn([in_dim, out_dim])

        self.params["b"] = np.zeros([1, out_dim])

    def forward(self, x):
        self.buffer["x"] = x
        return np.dot(x, self.params["w"]) + self.params["b"]

    def backward(self, dout):
        self.grad["w"] = np.dot(self.buffer["x"].transpose(), dout)
        self.grad["b"] = np.sum(dout, axis=0, keepdims=True)
        din = np.dot(dout, self.params["w"].transpose())
        return din
