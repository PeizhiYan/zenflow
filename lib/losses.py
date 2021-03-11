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



def mean_squared_loss(y_pred, y_targ):
    """mean squared error loss function""" 
    #  (1/2) || y_pred - y_targ ||^2
    # || . || is the Euclidean norm
    return (1. / 2) * np.square(y_targ - y_pred)

def d_mean_squared_loss(y_pred, y_targ):
    """derivative of loss function w.r.t. y_targ"""
    return y_pred - y_targ

def binary_cross_entropy(y_pred, y_targ):
    """binary cross-entropy loss"""
    # y_targ and y_pred suppose to have shape [n,1]
    return -(np.multiply(y_targ, np.log(y_pred)) + np.multiply(1-y_targ, np.log(1-y_pred)))

def d_binary_cross_entropy(y_pred, y_targ):
    """derivative of binary cross-entropy loss"""
    return (y_pred - y_targ) / (y_pred * (1 - y_pred))


