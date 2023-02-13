import numpy as np


def get_metric(name):
    act_fnc = [Accuracy()]
    for func in act_fnc:
        if func.name == name.lower() or func.short == name.lower():
            return func
    return None


class Accuracy():
    def __init__(self):
        self.type = 'metric'
        self.name = 'accuracy'
        self.short = 'acc'

    def __call__(self, y_pred, y_true):
        return (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)).mean()*100
