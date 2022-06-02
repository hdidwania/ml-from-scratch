"""
Implementation of linear layer
"""

import numpy as np

from ml.layers.base import LearnableLayer


class Linear(LearnableLayer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.params["w"] = np.random.randn(in_dim, out_dim)
        self.params["b"] = np.random.randn(1, out_dim)

    def forward(self, x):
        self.buffer["x"] = x
        return np.dot(x, self.params["w"]) + self.params["b"]

    def backward(self, dout):
        self.grad["w"] = np.dot(self.buffer["x"].transpose(), dout)
        self.grad["b"] = np.sum(dout, axis=0, keepdims=True)
        din = np.dot(dout, self.params["w"].transpose())
        return din
