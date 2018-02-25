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

#ifndef DATA_CUH
#define	DATA_CUH

#include <vector>
#include <algorithm>
#include "util.cuh"

class CPUData {
protected:
    MatrixV* _data;
    void assertDimensions() {
        assert(_data->size() > 0);
        for (int i = 1; i < _data->size(); i++) {
            assert(_data->at(i-1)->getNumCols() == _data->at(i)->getNumCols());
            if (_data->at(i-1)->isTrans() != _data->at(i)->isTrans() && _data->at(i)->getNumElements() < 2) {
                _data->at(i)->setTrans(_data->at(i-1)->isTrans());
            }
            assert(_data->at(i-1)->isTrans() == _data->at(i)->isTrans());
        }
        assert(_data->at(0)->getNumCols() > 0);
    }
public:
    typedef typename MatrixV::iterator T_iter;
    // Cases in columns, but array may be transposed
    // (so in memory they can really be in rows -- in which case the array is transposed
    //  during the copy to GPU).
    CPUData(PyObject* pyData) {
        _data = getMatrixV(pyData);
        assertDimensions();
    }
    
    CPUData(MatrixV* data) : _data(data) {
        assertDimensions();
    }

    ~CPUData() {
        for (T_iter it = _data->begin(); it != _data->end(); ++it) {
            delete *it;
        }
        delete _data;
    }
    
    Matrix& operator [](int idx) const {
        return *_data->at(idx);
    }
    
    int getSize() const {
        return _data->size();
    }
    
    MatrixV& getData() const {
        return *_data;
    }
    
    Matrix& getData(int i) const {
        return *_data->at(i);
    }
    
    bool isTrans() const {
        return _data->at(0)->isTrans();
    }

    int getNumCases() const {
        return _data->at(0)->getNumCols();
    }
};

class DataProvider {
protected:
    CPUData* _hData;
    NVMatrixV _data;
    int _minibatchSize;
public:
    DataProvider(int minibatchSize);
    void setData(CPUData&);
    void clearData();
    CPUData& getMinibatch(int idx);
    CPUData& getDataSlice(int startCase, int endCase);
    int getNumMinibatches();
    int getMinibatchSize();
    int getNumCases();
};

#endif	/* DATA_CUH */

