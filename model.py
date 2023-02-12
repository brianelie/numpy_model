import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from loss import get_loss
from metrics import get_metric
from optimizer import get_optimizer
from layers import Input

import time

import tensorflow as tf

class Model:
    # Code base from https://medium.com/analytics-vidhya/neural-network-mnist-classifier-from-scratch-using-numpy-library-94bbcfed7eae
    def __init__(self, layers):
        self.layers = layers
        
        if self.layers[0].input_shape is None:
            sys.exit('No input shape given')

        self.input_shape = self.layers[0].input_shape

        output_shape = self.input_shape

        layer_names = []
        for layer in layers:
            i = 0
            name = f'{layer.name}_{i}'
            while name in layer_names:
                i += 1
                name = f'{layer.name}_{i}'
            layer_names.append(name)
            layer.name = name
            
            output_shape = layer.get_output_shape(output_shape)
            layer.init_weights()
            
        self.output_shape = output_shape
            

    def compile(self, optim='sgd',metrics='', loss=''):
        if type(metrics) == str and metrics:
            self.metrics = [get_metric(metrics)]
        elif metrics:
            self.metrics = [get_metric(name) for name in metrics]
        else:
            self.metrics = []
            
        self.optim = get_optimizer(optim)

        self.loss_fn = get_loss(loss)

    def summary(self):
        print('Model Summary')
        total_params = 0
        for layer in self.layers:
            total_params += layer.summary()
        print(f'Total Parameters:{total_params}')

    def feedforward(self, x, training):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backprop(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self):
        for layer in self.layers:
            if layer.type == 'layer' and layer.trainable:
                layer.update_weights(self.optim)

    def train(self, train_data, batch_size=64, lr=1e-3,  epochs=50, val_data=None, val_patience=None):
        self.optim.initialize(lr, self.layers)
        n_batches = train_data.num_samples//batch_size-1

        self.results = pd.DataFrame(columns=['loss'])
        if val_data:
            self.results['val_loss'] = None
        for metric in self.metrics:
            self.results[metric.short] = None
            if val_data:
                self.results[f'val_{metric.short}'] = None

        if val_patience and not val_data:
            sys.exit('Validation patience entered without validation data.')
        elif val_patience:
            self.best_epoch = 1

        for epoch in range(1, epochs+1):
            train_data.shuffle()
            batched = train_data.batch(batch_size)
            loss = 0
            metrics = np.zeros(len(self.metrics))
            with tqdm(total=n_batches, unit=' batches') as tepoch:
                tepoch.set_description_str(f'Epoch {epoch}')
                for num, (x, y) in enumerate(batched):
                    batch_x = x
                    batch_y = np.squeeze(y)
                    # s = time.time()
                    y_pred = self.feedforward(batch_x, training=True)
                    # e = time.time() - s
                    # print(f'Forward: {e}')

                    loss += self.loss_fn.forward(y_pred, batch_y)/x.shape[0]
                    grad = self.loss_fn.backward(y_pred, batch_y)

                    # s = time.time()
                    self.backprop(grad)
                    # e = time.time() - s
                    # print(f'Backward: {e}')
                    self.update_weights()

                    self.results.loc[epoch, 'loss'] = loss/(num+1)
                    for i in range(len(self.metrics)):
                        metrics[i] += self.metrics[i](y_pred, batch_y)
                        self.results.loc[epoch, metric.short] = metrics[i]/(num+1)

                    tepoch.set_postfix(self.results.loc[epoch].to_dict())
                    tepoch.update(1)

                if val_data:
                    val_loss = 0
                    val_metrics = np.zeros(len(self.metrics))
                    for num, (val_x, val_y) in enumerate(val_data.batch(batch_size)):
                        val_y = np.squeeze(val_y)
                        val_y_pred = self.feedforward(val_x, training=False)
                        val_loss += self.loss_fn.forward(val_y_pred, val_y)/val_x.shape[0]

                        self.results.loc[epoch, 'val_loss'] = val_loss/(num+1)
                        for metric in self.metrics:
                            val_metrics[i] += self.metrics[i](val_y_pred, val_y)
                            self.results.loc[epoch, f'val_{metric.short}'] = val_metrics[i]/(num+1)

                tepoch.set_postfix(self.results.loc[epoch].to_dict())

                if val_patience and not self.early_stopping(val_patience, epoch, val_loss):
                    for layer in self.layers:
                        if layer.trainable:
                            layer.set_best_weights()
                    break
        return self.results

    def early_stopping(self, val_patience, epoch, val_loss):
        if not val_patience:
            return True
        if epoch == 1:
            self.val_best = val_loss
        elif val_loss < self.val_best:
            self.val_best = val_loss
            self.best_epoch = epoch
            for layer in self.layers:
                if layer.trainable:
                    layer.save_weights()
        elif (epoch - self.best_epoch) >= val_patience:
            print(
                f'Stopping on early stopping, best epoch: {self.best_epoch} with val loss: {self.val_best:.3f}')
            return False
        return True

    def save_state(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'layers':self.layers,'optim':self.optim}, f)

    def load_state(self, path):
        with open(path, 'rb') as f:
            save_state = pickle.load(f)

        self.layers = save_state['layers']
        self.optim = save_state['optim']

    def plot_results(self):
        plt.figure(dpi=125)
        plt.plot(self.results['loss'], label='Loss', color='y')
        if 'val_loss' in self.results.columns:
            plt.plot(self.results['val_loss'], label='Val Loss', color='b')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        for metric in self.metrics:
            plt.figure(dpi=125)
            plt.plot(self.results[metric.short], label=metric.name, color='y')
            if f'val_{metric.short}' in self.results.columns:
                plt.plot(self.results[f'val_{metric.short}'],
                         label=f'Val {metric.name}', color='b')
            plt.xlabel("Epochs")
            plt.ylabel(metric.name)
            plt.legend()
            plt.show()

    def test(self, test_data):
        x, y = test_data.get_data()
        y_pred = self.feedforward(x, training=False)
        for metric in self.metrics:
            value = metric(y_pred, y)
            print(f'{metric.name}: {value:.2f}%')
