class BaseOptimizer:
    def __init__(self):
        pass

    def update(self):
        pass


class SGD(BaseOptimizer):
    def __init__(self, lr):
        self.lr = lr

    def update(self, w, dw):
        return w - self.lr * dw
