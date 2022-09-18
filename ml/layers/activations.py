"""
Implementation of activation functions
"""

import numpy as np

from ml.layers.base import ActivationLayer


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sigmoid_x = self.sigmoid(x)
        self.buffer["sigmoid_x"] = sigmoid_x
        return sigmoid_x

    def backward(self, dy):
        sigmoid_x = self.buffer["sigmoid_x"]
        return sigmoid_x * (1 - sigmoid_x) * dy

    def sigmoid(self, x):
        return np.where(
            x >= 0,
            1 / (1 + np.nan_to_num(np.exp(-x))),
            np.nan_to_num(np.exp(x)) / (1 + np.nan_to_num(np.exp(x))),
        )


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mask = np.where(x > 0, 1, 0)
        self.buffer["mask"] = mask
        return x * mask

    def backward(self, dy):
        mask = self.buffer["mask"]
        return dy * mask
