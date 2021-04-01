"""
MIT License

Copyright (c) 2021 Peizhi Yan (Matthew)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import


from lib.zenmath import *
from lib.losses import *
from lib.layers import dense_layer

# this dictionary defines the derivative of loss functions
derivative_loss = {
    mean_squared_loss: d_mean_squared_loss,
    binary_cross_entropy: d_binary_cross_entropy
}

class sequential:
    """sequential model"""

    def __init__(self, loss_function):
        self.layers = []
        self.loss = loss_function
        self.d_loss = derivative_loss[self.loss]

    def summary(self):
        """summarize the model"""
        if len(self.layers) == 0:
            print('Empty model')
            return
        print('=====================================================')
        print('Input dimension: [?, {}]'.format(self.layers[0].W.shape[0]))
        print('-----------------------------------------------------')
        n_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            n_params += layer.W.shape[0] * layer.W.shape[1]
            print('Dense layer {}'.format(i) + \
            '\t shape: [{}, {}]'.format(layer.W.shape[0],layer.W.shape[1]) + \
            '\t activation: ' + layer.activation.__name__
            )
        print('-----------------------------------------------------')
        print('Output dimension: [?, {}]'.format(self.layers[-1].W.shape[1]))
        print('=====================================================')
        print('Loss function: ' + self.loss.__name__)
        print('Number of parameters: {:,}'.format(n_params))
        print('=====================================================')

    def add_layer(self, layer):
        """add a layer to the sequential model"""
        self.layers.append(layer)

    def model_loss(self, X, Y):
        """compute the model loss on given data"""
        return np.mean(self.loss(self.predict(X), Y))

    def predict(self, X, return_score=True):
        """make predictions"""
        Z = X # hidden layer output (start as input layer)
        for layer in self.layers:
            # pass the data through each layer
            Z = self.forward(layer, Z, layer.activation)
        if return_score:
            # can also be used for regression output
            # when computing loss, also use the raw scores
            return Z
        else:
            # categorical prediction
            if Z.shape[1] == 1:
                return 1 * (Z > 0.5) # binary format
            else:
                return np.argmax(Z) # one-hot format
            
    def auto_grad(self, X, Y):
        """compute gradient for each layer"""
        # Y: the target outputs (labels)
        """forward pass"""
        Z = X # hidden layer output (start as input layer)
        for layer in self.layers:
            # pass the data through each layer
            Z = self.forward(layer, Z, layer.activation)
        """back propagation"""
        output_layer = self.layers[-1] # get output layer
        if self.loss is binary_cross_entropy:
            delta = output_layer.Z - Y
        else:
            loss_derivative = self.d_loss(output_layer.Z, Y)
            delta = loss_derivative * output_layer.d_activation(output_layer.XW) 
        output_layer.delta = delta
        output_layer.gradient = output_layer.X.T @ output_layer.delta
        skip_last = True
        W_next = output_layer.W # next layer's weight
        for layer in self.layers[::-1]:
            # reverse traverse each layer
            if skip_last:
                # skip the output layer
                skip_last = False
                continue
            delta = (delta @ W_next.T) * layer.d_activation(layer.XW)
            W_next = layer.W
            layer.delta = delta
            layer.gradient = layer.X.T @ layer.delta

    def update_step(self, learning_rate):
        """update the weights of each layer"""
        for layer in self.layers:
            layer.W = layer.W - learning_rate * layer.gradient
            layer.gradient = layer.gradient * 0 # clear gradient        
    
    def forward(self, layer, X, activation):
        """forward pass to a single layer"""
        layer.X = X # set layer input
        layer.XW = layer.X @ layer.W # set layer XW
        layer.Z = activation(layer.XW) # set layer output
        return layer.Z

    def save(self, fpath):
        """save layers as a single numpy file"""
        np.save(fpath, self.layers)

    def load(self, fpath):
        """load layers from the numpy file"""
        self.layers = list(np.load(fpath, allow_pickle=True))


