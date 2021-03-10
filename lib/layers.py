# Copyright 2021 Peizhi Yan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
    




