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

#ifndef UTIL_H
#define	UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <string>
#include <Python.h>
#include "../../nvmatrix/include/nvmatrix.cuh"
#include "../../util/include/matrix.h"


#define PASS_TYPE                   uint
#define PASS_TRAIN                  0x1
#define PASS_TEST                   0x2
#define PASS_GC                     0x4
#define PASS_MULTIVIEW_TEST         (PASS_TEST | 0x8)
#define PASS_MULTIVIEW_TEST_START   (PASS_MULTIVIEW_TEST | 0x10)
#define PASS_MULTIVIEW_TEST_END     (PASS_MULTIVIEW_TEST | 0x20)
#define PASS_FEATURE_GEN            0x40

#define HAS_FLAG(f, x)              (((x) & (f)) == (f))
#define IS_MULTIVIEW_TEST(x)        HAS_FLAG(PASS_MULTIVIEW_TEST, x)
#define IS_MULTIVIEW_TEST_START(x)  HAS_FLAG(PASS_MULTIVIEW_TEST_START, x)
#define IS_MULTIVIEW_TEST_END(x)    HAS_FLAG(PASS_MULTIVIEW_TEST_END, x)
#define IS_TEST(x)                  HAS_FLAG(PASS_TEST, x)
#define IS_TRAIN(x)                 HAS_FLAG(PASS_TRAIN, x)

// For gradient checking
#define GC_SUPPRESS_PASSES          false
#define GC_REL_ERR_THRESH           0.02

#ifdef DO_PRINT
#define PRINT(x, args...) printf(x, ## args);
#else
#define PRINT(x, args...) ;
#endif

/*
 * Generates a random floating point number in the range 0-1.
 */
#define randf                       ((float)rand() / RAND_MAX)

//typedef std::vector<Matrix*> MatrixV;
//typedef std::vector<NVMatrix*> NVMatrixV;
typedef std::map<std::string,std::vector<double>*> CostMap;
typedef std::map<std::string,double> CostCoeffMap;
typedef std::vector<double> doublev;
typedef std::vector<float> floatv;
typedef std::vector<int> intv;
typedef std::vector<std::string> stringv;
typedef std::set<int> seti;
typedef std::vector<PyObject*> PyObjectV;

stringv* getStringV(PyObject* pyList);
floatv* getFloatV(PyObject* pyList);
intv* getIntV(PyObject* pyList);
MatrixV* getMatrixV(PyObject* pyList);
MatrixV* getMatrixV(PyObject* pyList, int len);
int* getIntA(PyObject* pyList);

int pyDictGetInt(PyObject* dict, const char* key);
intv* pyDictGetIntV(PyObject* dict, const char* key);
std::string pyDictGetString(PyObject* dict, const char* key);
float pyDictGetFloat(PyObject* dict, const char* key);
floatv* pyDictGetFloatV(PyObject* dict, const char* key);
Matrix* pyDictGetMatrix(PyObject* dict, const char* key);
MatrixV* pyDictGetMatrixV(PyObject* dict, const char* key);
int* pyDictGetIntA(PyObject* dict, const char* key);
stringv* pyDictGetStringV(PyObject* dict, const char* key);
bool pyDictHasKey(PyObject* dict, const char* key);
PyObjectV* pyDictGetValues(PyObject* dict);

template<typename T> std::string tostr(T n);
template<typename T> void shuffleVector(std::vector<T>& v, int start, int end);
template<class T> void deleteElements(std::vector<T*>& v);
template<class T> void deleteElements(std::vector<T*>& v, bool deleteContainer);

template<class T>
int indexOf(std::vector<T>& v, T e) {
    int i = 0;
//    typename vector<T>::iterator it2 = v.begin();
    for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
        if (*it == e) {
            return i;
        }
        ++i;
    }
    return -1;
}

std::vector<int>& getDeviceCPUs(int deviceID);

template<typename K, typename V> std::set<K> getKeys(std::map<K,V>& m) {
    std::set<K> s;
    for (typename std::map<K,V>::const_iterator it = m.begin(); it != m.end(); ++it) {
        s.insert(it->first);
    }
    return s;
}

struct LayerIDComparator {
    bool operator()(PyObject* i, PyObject* j) {
        return pyDictGetInt(i, "id") < pyDictGetInt(j, "id");
    }
};

#endif	/* UTIL_H */

