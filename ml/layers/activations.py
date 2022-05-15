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
        return 1 / (1 + np.exp(-x))
