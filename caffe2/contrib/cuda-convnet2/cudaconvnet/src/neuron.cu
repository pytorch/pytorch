/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/neuron.cuh"
#include "../include/util.cuh"

using namespace std;

Neuron& Neuron::makeNeuron(PyObject* neuronDict) {
    std::string type = pyDictGetString(neuronDict, "type");
    PyObject* neuronParamsDict = PyDict_GetItemString(neuronDict, "params");
    
    if (type == "relu") {
        return *new ReluNeuron();
    }
    
    if (type == "drelu") {
        return *new DoubleReluNeuron(pyDictGetFloat(neuronParamsDict, "a"));
    }
    
    if (type == "softrelu") {
        return *new SoftReluNeuron();
    }
    
    if (type == "brelu") {
        return *new BoundedReluNeuron(pyDictGetFloat(neuronParamsDict, "a"));
    }

    if (type == "abs") {
        return *new AbsNeuron();
    }

    if (type == "logistic") {
        return *new LogisticNeuron();
    }
    
    if (type == "tanh") {
        return *new TanhNeuron(pyDictGetFloat(neuronParamsDict, "a"), pyDictGetFloat(neuronParamsDict, "b"));
    }
    
    if (type == "square") {
        return *new SquareNeuron();
    }
    
    if (type == "sqrt") {
        return *new SqrtNeuron();
    }
    
    if (type == "linear") {
        return *new LinearNeuron(pyDictGetFloat(neuronParamsDict, "a"), pyDictGetFloat(neuronParamsDict, "b"));
    }

    if (type == "log") {
        return *new LogNeuron(pyDictGetFloat(neuronParamsDict, "a"));
    }

    if (type == "ident") {
        return *new Neuron();
    }
    
    throw std::string("Unknown neuron type: ") + type;
}
