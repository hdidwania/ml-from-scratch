"""
Base classes for different layers
"""


class LearnableLayer:
    def __init__(self):
        self.type = "learnable"
        self.params = dict()
        self.grad = dict()
        self.buffer = dict()

    def forward(self):
        pass

    def backward(self):
        pass

    def get_params(self):
        return self.params


class ActivationLayer:
    def __init__(self):
        self.type = "activation"
        self.buffer = dict()

    def forward(self):
        pass

    def backward(self):
        pass
