import numpy as np


class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_classes = len(np.unique(self.y))
        self.num_samples = self.x.shape[0]
        self.batch_size = self.num_samples

        if self.x.ndim == 2:
            self.num_features = self.x.shape[1]
        else:
            self.num_features = np.prod(self.x[1:])

    def one_hot(self):
        self.y = np.eye(self.num_classes)[self.y]

    def normalize(self, min=None, max=None):
        # Normalize values between 0 and 1
        if min and max:
            self.x = (self.x - min)/(max-min)
            return
        self.x = (self.x - np.min(self.x))/(np.max(self.x)-np.min(self.x))
        return np.min(self.x), np.max(self.x)

    def flatten(self):
        self.x = np.reshape(self.x, (self.x.shape[0], -1))
        if not self.num_features:
            self.num_features = self.x.shape[1]

    def batch(self, batch_size):
        """
        :param batch_size:  size of each group
        :return:            Generator for batches of data.
        """
        self.batch_size = batch_size
        for i in range(0, len(self.x), batch_size):
            yield self.x[i:i+batch_size], self.y[i:i+batch_size]

    def shuffle(self):
        idx = np.random.permutation(self.num_samples)
        self.x = self.x[idx]
        self.y = self.y[idx]

    def get_data(self):
        return self.x, self.y
    
    def get_shape(self):
        return self.x.shape[1:]
