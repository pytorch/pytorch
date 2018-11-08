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

#include <Python.h>
#include <arrayobject.h>
#include <helper_cuda.h>
#include "../include/util.cuh"

using namespace std;

stringv* getStringV(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    stringv* vec = new stringv(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(std::string(PyString_AS_STRING(PyList_GET_ITEM(pyList, i))));
    }
    return vec;
}

floatv* getFloatV(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    floatv* vec = new floatv(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(PyFloat_AS_DOUBLE(PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

intv* getIntV(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    intv* vec = new intv(); 
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        vec->push_back(PyInt_AS_LONG(PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

int* getIntA(PyObject* pyList) {
    if (pyList == NULL) {
        return NULL;
    }
    int* arr = new int[PyList_GET_SIZE(pyList)];
    for (int i = 0; i < PyList_GET_SIZE(pyList); i++) {
        arr[i] = PyInt_AS_LONG(PyList_GET_ITEM(pyList, i));
    }
    return arr;
}

MatrixV* getMatrixV(PyObject* pyList) {
    return getMatrixV(pyList, PyList_GET_SIZE(pyList));
}

MatrixV* getMatrixV(PyObject* pyList, int len) {
    if (pyList == NULL) {
        return NULL;
    }
    MatrixV* vec = new MatrixV(); 
    for (int i = 0; i < len; i++) {
        vec->push_back(new Matrix((PyArrayObject*)PyList_GET_ITEM(pyList, i)));
    }
    return vec;
}

PyObjectV* pyDictGetValues(PyObject* dict) {
    PyObjectV* pov = new PyObjectV();
    PyObject* valuesList = PyDict_Values(dict);
    int numValues = PyList_GET_SIZE(valuesList);

    for (int i = 0; i < numValues; i++) {
        pov->push_back(PyList_GET_ITEM(valuesList, i));
    }
    Py_DECREF(valuesList);
    return pov;
}

int pyDictGetInt(PyObject* dict, const char* key) {
    return PyInt_AS_LONG(PyDict_GetItemString(dict, key));
}

intv* pyDictGetIntV(PyObject* dict, const char* key) {
    return getIntV(PyDict_GetItemString(dict, key));
}

int* pyDictGetIntA(PyObject* dict, const char* key) {
    return getIntA(PyDict_GetItemString(dict, key));
}

std::string pyDictGetString(PyObject* dict, const char* key) {
    return std::string(PyString_AS_STRING(PyDict_GetItemString(dict, key)));
}

float pyDictGetFloat(PyObject* dict, const char* key) {
    return PyFloat_AS_DOUBLE(PyDict_GetItemString(dict, key));
}

floatv* pyDictGetFloatV(PyObject* dict, const char* key) {
    return getFloatV(PyDict_GetItemString(dict, key));
}

Matrix* pyDictGetMatrix(PyObject* dict, const char* key) {
    return new Matrix((PyArrayObject*)PyDict_GetItemString(dict, key));
}

MatrixV* pyDictGetMatrixV(PyObject* dict, const char* key) {
    return getMatrixV(PyDict_GetItemString(dict, key));
}

stringv* pyDictGetStringV(PyObject* dict, const char* key) {
    return getStringV(PyDict_GetItemString(dict, key));
}

bool pyDictHasKey(PyObject* dict, const char* key) {
    PyObject* str = PyString_FromString(key);
    bool b = PyDict_Contains(dict, str);
    Py_DECREF(str);
    return b;
}

template<typename T>
void shuffleVector(vector<T>& v, int start, int end) {
    const int len = end - start;
    for (int i = 0; i < len*5; ++i) {
        int r1 = start + rand() % len;
        int r2 = start + rand() % len;
        int tmp = v[r1];
        v[r1] = v[r2];
        v[r2] = tmp;
    }
}

template<class T>
std::string tostr(T n) {
    ostringstream result;
    result << n;
    return result.str();
}

template<class T>
void deleteElements(vector<T*>& v) {
    deleteElements(v, false);
}

template<class T>
void deleteElements(vector<T*>& v, bool deleteContainer) {
    for (typename vector<T*>::const_iterator it = v.begin(); it != v.end(); ++it) {
        delete *it;
    }
    if (deleteContainer) {
        delete &v;
    }
}

static Lock deviceCPULock;
static std::map<int, std::vector<int> > deviceCPUs;

std::vector<int>& getDeviceCPUs(int deviceID) {
    deviceCPULock.acquire();
    if (deviceCPUs.count(deviceID) == 0 && deviceID >= 0) {
        struct cudaDeviceProp props;
        checkCudaErrors(cudaGetDeviceProperties(&props, deviceID));
        char pciString[13];

        sprintf(pciString, "%04x", props.pciDomainID);
        pciString[4] = ':';
        sprintf(pciString + 5, "%02x", props.pciBusID);
        pciString[7] = ':';
        sprintf(pciString + 8, "%02x", props.pciDeviceID);
        pciString[10] = '.';
        pciString[11] = '0';
        pciString[12] = 0;
        std::string path = std::string("/sys/bus/pci/devices/") + std::string(pciString) + "/local_cpulist";
        ifstream f(path.c_str());

        if (f.is_open()) {
            std::string cpuString;
            while (getline(f, cpuString, ',')) {
                int start, end;
                int found = sscanf(cpuString.c_str(), "%d-%d", &start, &end);
                end = found == 1 ? start : end;
                if (found > 0) {
                    for (int i = start; i <= end; ++i) {
                        deviceCPUs[deviceID].push_back(i);
                    }
                } 
            }
            f.close();
        } else {
            printf("Unable to open %s\n", path.c_str());
        }
    }
    vector<int>& ret = deviceCPUs[deviceID];
    deviceCPULock.release();
    return ret;
}

template void shuffleVector<int>(std::vector<int>& v, int start, int end);
template std::string tostr<int>(int n);
template void deleteElements<NVMatrix>(std::vector<NVMatrix*>& v, bool deleteContainer);
