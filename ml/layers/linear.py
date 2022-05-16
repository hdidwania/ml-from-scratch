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
        self.buffer["x"]
        return np.dot(x, self.params["w"]) + self.params["b"]
