"""
Implementation of various loss functions
"""
import numpy as np

class LossBase:
    def __init__(self):
        self.buffer = dict()

    def forward(self):
        pass

    def backward(self):
        pass

class MSELoss(LossBase):
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
