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

import numpy as np

from lib.zenmath import *

# this dictionary defines the derivative of activation functions
derivative_activation = {
    sin: d_sin,
    relu: d_relu,
    sigmoid: d_sigmoid,
    linear: d_linear
}

class dense_layer:
    """dense layer class"""
    # Reference: https://sudeepraja.github.io/Neural/

    def __init__(self, in_dim, out_dim, activation=linear, initialize='random_uniform'):
        self.in_dim = in_dim    # input dimension
        self.out_dim = out_dim  # output dimension
        self.activation = activation
        self.d_activation = derivative_activation[self.activation] # the derivative of activation function
        if initialize == 'random_uniform':
            self.W = self.get_weights_random_uniform(_min=-1, _max=1)
        self.X = 0        # layer input
        self.XW = 0       # 
        self.Z = 0        # layer output
        self.delta = 0    # we use this term to prevent redundant computation
        self.gradient = 0 # the gradient of weights (for gradient descent)

    def get_weights_random_uniform(self, _min, _max):
        """ generate random weights (uniform distribution) """
        # _min: minimum value
        # _max: maximum value
        return (_max - _min) * np.reshape(np.random.rand(self.in_dim * self.out_dim), [self.in_dim, self.out_dim]) + _min
    




