import numpy as np

def get_activation(name):
    act_fnc = [Softmax(), Sigmoid(), Relu(), Tanh()]
    for func in act_fnc:
        if func.name == name.lower():
            return func
    return None

class Layer(object):
    def __init__(self, activation = '', input_shape = None):
        self.type = ''
        self.name = ''
        self.trainable = False
        if activation:
            self.act = get_activation(activation)
        else:
            self.act = None
        self.weight_shape = 0
        self.bias_shape = 0
        self.weights = None
        self.biases = None
        if input_shape:
            if len(input_shape) < 3: 
                input_shape = np.append(input_shape, 1)
        
            # Add a batch to the shape
            input_shape = np.insert(input_shape, 0, 0, axis=0)
        self.input_shape = input_shape
        
    def init_weights(self):
        self.weights = np.random.normal(
            scale=0.5, size=self.weight_shape)
        self.biases = np.zeros(self.bias_shape)
        self.save_weights()
        
    def save_weights(self):
        self.best_weights = self.weights
        self.best_biases = self.biases

    def set_best_weights(self):
        self.weights = self.best_weights
        self.biases = self.best_biases

    def update_weights(self, optim):
        self.weights -= optim(self.name, self.grad_w, bias=False)
        self.biases -= optim(self.name, self.grad_b, bias=True)

    def get_weights(self):
        return self.weights, self.biases

    def get_params(self):
        if self.weights is not None:
            return self.weights.size + self.biases.size  
        else:
            return 0
        
    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return self.output_shape
        
    def summary(self):
        params = self.get_params()
        print(f'{self.name}: Activation:{self.act} Parameters:{params}')
        return params
    
    def forward(self, x, training=True):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError      
    
class Input(Layer):
    def __init__(self, input_shape):
        super().__init__(activation='')
        self.input_shape = input_shape
        
class Conv2D(Layer):
    # https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py
    def __init__(self, num_filters = 1, kernel_size=3, padding=0, strides=1, activation='',name=None, input_shape=None):
        # The weights are including each of the filters for the last layer
        super().__init__(activation, input_shape)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.trainable = True
        self.type = 'layer'
        if name:
            self.name = name
        else:
            self.name = 'conv2d'
            
        self.weights = None
        self.biases = None
            
    def forward(self, x, training=True):          
        output = np.zeros((x.shape[0], *self.output_shape[1:]))
        
        if x.ndim < 4:
            x = x[:,:,:,np.newaxis]
        
        if training:
            self.old_x = x
            
        if self.padding > 0:
            output = np.pad(output, pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
                 
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                i_end = i + self.kernel_size
                j_end = j + self.kernel_size
                output[:, i, j, :] = np.einsum(
                    'bijk,ijkf->bf', x[:, i:i_end, j:j_end, :], self.weights)
                
        output = output + self.biases          
        
        if self.act:
            return self.act.forward(output, training)
        return output
    
    def backward(self, grad):
        if self.act:
            grad = self.act.backward(grad)
            
        if self.padding > 0:
            self.old_x = np.pad(self.old_x, pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        
        _, h_out, w_out, _ = grad.shape
        n, h_in, w_in, _ = self.old_x.shape
        h_f, w_f, _, _ = self.weights.shape
        
        self.grad_w = np.zeros(self.weights.shape)
        self.grad_b = grad.sum(axis=(0,1,2))/n
        
        output = np.zeros_like(self.old_x, dtype=np.float64)
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.strides
                h_end = h_start + h_f
                w_start = j * self.strides
                w_end = w_start + w_f
                
                output[:, h_start:h_end, w_start:w_end, :] += np.einsum('ijcf,bf->bijc', self.weights,grad[:,i,j,:])
                
                slice_x = self.old_x[:, h_start:h_end, w_start:w_end, :]
                slice_grad = grad[:,i,j,:]
                self.grad_w += np.einsum('bijc,bf->ijcf',slice_x, slice_grad)

        return output
    
    
    def summary(self):
        params = self.get_params()
        if self.act:
            act_line = f'Activation: {self.act.name}'
        else:
            act_line = ''
        print(f'{self.name}: {self.num_filters} x {self.kernel_size} x {self.kernel_size} {act_line} Parameters:{params}')
        return params
    
    def get_output_shape(self, input_shape):
        self.input_shape = input_shape

        # batch, height, width
        n, h_in, w_in = input_shape[:3]

        h_f = self.kernel_size
        w_f = self.kernel_size
        n_f = self.num_filters

        h_out = (h_in - h_f + 2*self.padding) // self.strides + 1
        w_out = (w_in - w_f + 2*self.padding) // self.strides + 1
        # batch, height, width, number of filters
        self.output_shape = n, h_out, w_out, n_f
            
        return self.output_shape
    
    def init_weights(self):
        channels = self.input_shape[-1]
        self.weight_shape = [self.kernel_size, self.kernel_size, channels, self.num_filters]
        self.bias_shape = self.num_filters
        super().init_weights()
    
class Flatten(Layer):
    def __init__(self, name = None, input_shape=()):
        super().__init__(input_shape=input_shape)
        self.type='layer'
        self.name = name if name else 'flatten'
        self.shape = input_shape
        
    def forward(self, x, training=True):
        if training:
            self.shape = x.shape
        return np.reshape(x, (x.shape[0],-1))

    def backward(self, grad):
        return np.reshape(grad, (self.shape))
    
    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = np.prod(self.input_shape[1:])
        return self.output_shape
    
class MaxPooling(Layer):
    def __init__(self, pool_size = 2, strides = 2, name=None):
        super().__init__()
        self.type='layer'
        self.name = name if name else 'max_pool'
        self.pool_size = pool_size
        self.strides = strides
                        
    def forward(self, x, training = True):
        if training:
            self.old_x = x
        
        n, h, w, f = x.shape
        
        h_out = (h - self.pool_size) // self.strides + 1
        w_out = (w - self.pool_size) // self.strides + 1
        
        output = np.empty((n, h_out, w_out, f))
        
        for i in range(h_out):
            for j in range(w_out):
                h_s = i*self.strides
                h_e = h_s + self.pool_size
                w_s = j*self.strides
                w_e = w_s + self.pool_size
                slice_x = x[:, h_s:h_e, w_s:w_e, :]
                output[:,i,j,:] = np.max(slice_x,axis=(1,2))
        return output
    
    def backward(self, grad):
        output = np.zeros(self.old_x.shape)
        
        n, h, w, f = self.old_x.shape
        
        h_out = (h - self.pool_size) // self.strides + 1
        w_out = (w - self.pool_size) // self.strides + 1

        for i in range(h_out):
            for j in range(w_out):
                h_s = i*self.strides
                h_e = h_s + self.pool_size
                w_s = j*self.strides
                w_e = w_s + self.pool_size
                
                slice_x = self.old_x[:,h_s:h_e,w_s:w_e,:]
                max_x_pool = np.max(slice_x, axis=(1,2))
                mask = (slice_x == (max_x_pool)[:, None, None, :])
                output[:,h_s:h_e, w_s:w_e,:] += mask * (grad[:,i,j,:])[:, None, None, :]
        
        return output
                
    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        b, h, w, c = self.input_shape
        h_out = (h - self.pool_size) // self.strides + 1
        w_out = (w - self.pool_size) // self.strides + 1
        self.output_shape = (b, h_out, w_out, c)
        return self.output_shape
    
class AvgPooling(Layer):
    def __init__(self, pool_size=2, strides=2, name=None):
        super().__init__()
        self.type = 'layer'
        self.name = name if name else 'avg_pool'
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, x, training=True):
        if training:
            self.old_x = x

        n, h, w, f = x.shape

        h_out = (h - self.pool_size) // self.strides + 1
        w_out = (w - self.pool_size) // self.strides + 1

        output = np.empty((n, h_out, w_out, f))

        for i in range(h_out):
            for j in range(w_out):
                h_s = i*self.strides
                h_e = h_s + self.pool_size
                w_s = j*self.strides
                w_e = w_s + self.pool_size
                slice_x = x[:, h_s:h_e, w_s:w_e, :]
                output[:, i, j, :] = np.mean(slice_x, axis=(1, 2))
        return output

    def backward(self, grad):
        output = np.zeros(self.old_x.shape)

        n, h, w, f = self.old_x.shape

        h_out = (h - self.pool_size) // self.strides + 1
        w_out = (w - self.pool_size) // self.strides + 1

        for i in range(h_out):
            for j in range(w_out):
                h_s = i*self.strides
                h_e = h_s + self.pool_size
                w_s = j*self.strides
                w_e = w_s + self.pool_size
                
                output[:, h_s:h_e, w_s:w_e, :] = (grad[:, i, j, :])[:, None, None, :] / (self.pool_size*self.pool_size)

        return output

    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        b, h, w, c = self.input_shape
        h_out = (h - self.pool_size) // self.strides + 1
        w_out = (w - self.pool_size) // self.strides + 1
        self.output_shape = (b, h_out, w_out, c)
        return self.output_shape
    
class Dropout(Layer):
    def __init__(self, pdrop = 0.2, name = None):
        super().__init__(activation='')
        self.type='layer'
        self.name = name if name else 'dropout'
        self.pdrop = pdrop
        self.mask = None
    
    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.pdrop) / (1-self.pdrop)
            return self.mask*x
        else:
            return x
        
    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad*self.mask
         
class Dense(Layer):
    def __init__(self, n_out, activation='', name=None, input_shape=None):
        super().__init__(activation, input_shape)
        self.n_out = n_out
        self.type = 'layer'
        self.trainable = True
        if name:
            self.name = name
        else:
            self.name = 'dense'
            
        self.bias_shape = self.n_out

    def forward(self, x, training=True):
        output = np.dot(x, self.weights) + self.biases
        if training:
            self.old_x = x
        if self.act:
            return self.act.forward(output, training)
        return output

    def backward(self, grad):
        m = self.old_x.shape[0]
        if self.act:
            grad = self.act.backward(grad)
        self.grad_w = np.dot(self.old_x.T, grad)/m
        self.grad_b = grad.sum(axis=0)/m
        return np.dot(grad, self.weights.T)
    
    def summary(self):
        params = self.get_params()
        if self.act:
            act_line = f'Activation: {self.act.name}'
        else:
            act_line = ''
        print(f'{self.name}: {self.n_in} x {self.n_out} {act_line} Parameters:{params}')
        return params
    
    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.n_in = self.input_shape
        self.weight_shape = (self.n_in, self.n_out)
        self.output_shape = self.n_out
        return self.output_shape

class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'activation'
        self.name = 'relu'

    def forward(self, x, training=True):
        if training:
            self.old_x = np.copy(x)
        return np.maximum(x, 0)

    def backward(self, grad):
        return np.where(self.old_x > 0, grad, 0)

class Softmax(Layer):
    # https://stackoverflow.com/questions/36279904/softmax-derivative-in-numpy-approaches-0-implementation
    def __init__(self):
        super().__init__()
        self.type = 'activation'
        self.name = 'softmax'
    def forward(self, x, training):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = e / np.sum(e, axis=1, keepdims=True)
        if training:
            # self.x = x
            self.probs = probs
        return probs

    def backward(self, grad):
        return grad

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'sigmoid'

    def forward(self, x, training=True):
        x = np.clip(x, a_min=1e-8, a_max=None)
        output = 1/(1+np.exp(-x))
        if training:
            self.old_x = x
            self.old_y = output
        return output

    def backward(self, grad):
        return self.old_y*(1-self.old_y)*grad

class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'tanh'

    def forward(self, x, training=True):
        if training:
            self.old_x = x
        return np.tanh(x)

    def backward(self, grad):
        return grad*(1-np.tanh(self.old_x)**2)