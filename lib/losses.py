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


