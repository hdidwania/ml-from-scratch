class LearnableLayer:
    def __init__(self):
        self.params = dict()
        self.buffer = dict()

    def forward(self):
        pass

    def backward(self):
        pass

    def get_params(self):
        return self.params


class ActivationLayer:
    def __init__(self):
        self.buffer = dict()

    def forward(self):
        pass

    def backward(self):
        pass


class LossLayer:
    def __init__(self):
        self.buffer = dict()

    def forward(self):
        pass

    def backward(self):
        pass
