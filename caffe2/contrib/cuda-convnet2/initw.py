# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from python_util.gpumodel import *
import numpy as n
import numpy.random as nr

def get_src(filename):
    src = IGPUModel.load_checkpoint(filename)
    return src['model_state']['layers']
    
# Initialize weight matrix by copying weight matrix of given layer
def makew(name, idx, shape, params):
    src = get_src(params[0])
    return src[name]['weights'][idx]
    
# Initialize bias vector by copying bias vector of given layer
def makeb(name, shape, params):
    src = get_src(params[0])
    return src[name]['biases']
    
def concat(shape, src, src_layers, src_func):
    mat = n.empty(shape, dtype=n.single, order='F')
    start = 0
    for s in src_layers:
        m = src_func(src[s])
        mat[:,start:start+m.shape[1]] = m
        start += m.shape[1]
    return mat

# Initialize weight matrix by concatenating weight matrices of given layers
def makewcat(name, idx, shape, params):
    src, src_layers = get_src(params[0]), params[1:]
    return concat(shape, src, src_layers, lambda x: x['weights'][idx])
    
# Initialize bias vector by concatenating bias vectors of given layers
def makebcat(name, shape, params):
    src, src_layers = get_src(params[0]), params[1:]
    return concat(shape, src, src_layers, lambda x: x['biases'])

# Initialize bias vector from tuple input
def makeb_vec(name, shape, params):
    return n.array([n.single(x) for x in params], dtype=n.single).reshape((1, len(params)))
