import numpy as np

def get_optimizer(name):
    optim = [SGD(),SGDM(),Adam()]
    for func in optim:
        if func.name == name.lower():
            return func
    return None

class SGD():
    def __init__(self):
        self.type='optimizer'
        self.name='sgd'
        self.lr=1e-3
    
    def initialize(self, lr, layers):
        self.lr = lr
        
    def __call__(self, name, grad, bias=False):
        return self.lr*grad
    
class SGDM():
    def __init__(self, beta=0.9):
        self.type='optimizer'
        self.name='sgdm'
        self.lr=1e-3
        self.beta = beta
    
    def initialize(self, lr, layers):
        self.lr = lr
        self.layer_grads = {}
        for layer in layers:
            self.layer_grads[layer.name] = {'grad_w':0,'grad_b':0}
        
    def __call__(self, name, grad, bias=False):
        if bias:
            val = 'b'
        else:
            val = 'w'
        last_grad = self.layer_grads[name][f'grad_{val}']
        self.layer_grads[name][f'grad_{val}'] = grad
        vector = self.beta*grad + (1-self.beta)*last_grad
        return self.lr*(vector)
    
class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.type='optimizer'
        self.name='adam'
        self.lr=1e-3
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    
    def initialize(self, lr, layers):
        self.lr = lr
        self.layer_grads = {}
        for layer in layers:
            self.layer_grads[layer.name] = {'t':0,'m_w':0,'v_w':0,'m_b':0,'v_b':0}
        
    def __call__(self, name, grad, bias=False):
        if bias:
            val = 'b'
        else:
            val = 'w'
        self.layer_grads[name]['t'] += 1
        t = self.layer_grads[name]['t']
        m = self.beta1*self.layer_grads[name][f'm_{val}'] + (1-self.beta1)*grad
        self.layer_grads[name][f'm_{val}'] = m
        m = m/(1-self.beta1**t)
        v = self.beta2*self.layer_grads[name][f'v_{val}'] + (1-self.beta2)*grad**2
        self.layer_grads[name][f'v_{val}'] = v
        v = v/(1-self.beta2**t)
        return self.lr*m/(np.sqrt(np.abs(v))+self.epsilon)
