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

#include <algorithm>
#include <vector>
#include "../../util/include/matrix.h"
#include "../include/data.cuh"
#include "../include/timer.cuh"

using namespace std;

DataProvider::DataProvider(int minibatchSize) : 
    _minibatchSize(minibatchSize), _hData(NULL) {
}

void DataProvider::clearData() {
    delete _hData;
    _hData = NULL;
}

void DataProvider::setData(CPUData& hData) {
    // DataWorker calls clearData
    _hData = &hData;
    assert(_hData != NULL);
}

CPUData& DataProvider::getMinibatch(int idx) {
    assert(idx >= 0 && idx < getNumMinibatches());
    return getDataSlice(idx * _minibatchSize, (idx + 1) * _minibatchSize);
}

CPUData& DataProvider::getDataSlice(int startCase, int endCase) {
    assert(_hData != 0);
    assert(_hData->getNumCases() > 0);
    endCase = min(_hData->getNumCases(), endCase);
    // TODO: maintain these matrices, no point re-creating them all the time
    MatrixV& miniData = *new MatrixV();
    
    for (int i = 0; i < _hData->getData().size(); i++) {
        // NOTE: if hData is transposed, then the output minibatch matrix
        // can be a view. No need to allocate new CPU memory here. Might
        // want to look into optimizing that in the future, though it's 
        // unlikely to be a big deal.
        if (_hData->isTrans()) {
            miniData.push_back(&(*_hData)[i].sliceCols(startCase, endCase));
        } else {
            miniData.push_back(new Matrix());
            (*_hData)[i].sliceCols(startCase, endCase, *miniData.back());
        }
    }
    CPUData& cpuData = *new CPUData(&miniData);
    return *new CPUData(&miniData);
}

int DataProvider::getNumMinibatches() {
    assert(_hData != 0);
    assert(_hData->getNumCases() > 0);
    return DIVUP(_hData->getNumCases(), _minibatchSize);
}

int DataProvider::getMinibatchSize() {
    return _minibatchSize;
}

int DataProvider::getNumCases() {
    assert(_hData != 0);
    assert(_hData->getNumCases() > 0);
    return _hData->getNumCases();
}
