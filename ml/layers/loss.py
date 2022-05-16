"""
Implementation of various loss functions
"""
import numpy as np

from ml.layers.base import LossLayer


class MSELoss(LossLayer):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        self.buffer["true"] = true
        self.buffer["pred"] = pred
        return np.mean(np.square(true - pred)) / 2

    def backward(self):
        pred, true = self.buffer["pred"], self.buffer["true"]
        b, n = pred.shape
        return (pred - true) / (b * n)
