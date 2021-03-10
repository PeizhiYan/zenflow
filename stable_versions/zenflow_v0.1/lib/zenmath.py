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

def sin(x):
    """sin"""
    return np.sin(x)

def d_sin(x):
    """derivative of sin"""
    return np.cos(x)

def sigmoid(x):
    """sigmoid function"""
    # sigmoid(x) = 1 / (1 + e^(-x))
    return 1. / (1. + np.e**(-x))

def d_sigmoid(x):
    """derivative of sigmoid function"""
    return sigmoid(x) * (1. - sigmoid(x))

def relu(x):
    """ReLU"""
    # relu(x) = max(0, x)
    return x * (x > 0)

def d_relu(x):
    """derivative of ReLU"""
    return 1. * (x > 0)

def linear(x):
    """linear function"""
    return x

def d_linear(x):
    """derivative of linear function"""
    return 1. * (x * 0 == 0)





