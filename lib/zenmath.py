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





