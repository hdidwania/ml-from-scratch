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
        n_points = np.prod(pred.shape)
        return (pred - true) / n_points


class BCELoss(LossBase):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        self.buffer["true"] = true
        self.buffer["pred"] = pred
        sample_loss = -(
            true * np.nan_to_num(np.log(pred))
            + (1 - true) * np.nan_to_num(np.log(1 - pred))
        )
        return np.mean(sample_loss)

    def backward(self):
        pred, true = self.buffer["pred"], self.buffer["true"]
        n_points = np.prod(pred.shape)
        return -(true / pred - (1 - true) / (1 - pred)) / n_points
