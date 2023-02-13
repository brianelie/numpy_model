import numpy as np

def get_loss(name):
    act_fnc = [CrossEntropy(), MSE()]
    for func in act_fnc:
        if func.name == name.lower() or func.short == name.lower():
            return func
    return None


class CrossEntropy():
    def __init__(self):
        self.name = 'crossentropy'
        self.short = 'nll'

    def forward(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred+1e-8))

    def backward(self, y_pred, y_true):
        return y_pred - y_true

class MSE():
    def __init__(self):
        self.name = 'mse'
        self.short = 'mse'

    def forward(self, y_pred, y_true):
        return np.mean((y_pred-y_true)**2)

    def backward(self, y_pred, y_true):
        return (y_pred - y_true)