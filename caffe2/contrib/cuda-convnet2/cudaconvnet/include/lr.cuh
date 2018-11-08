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

#ifndef LR_CUH
#define LR_CUH

#include <string>
#include <vector>
#include <iostream>
#include <helper_cuda.h>
#include <assert.h>
#include <Python.h>
#include "util.cuh"
#include "../../nvmatrix/include/nvmatrix.cuh"
#include "../../util/include/matrix.h"

/*
 * The maximum learning rate is _baseRate.
 * The minimum learning rate is _baseRate / _tgtFactor.
 *
 * These classes define annealing schedules that interpolate between these
 * two extrema.
 */
class ParameterSchedule {
protected:
    double _baseRate;
public:
    ParameterSchedule(double base);
    virtual double getValue(double progress);
    double getBaseValue() const;
    virtual ~ParameterSchedule();

    static ParameterSchedule& make(PyObject* schedDict);
};

class LinearParameterSchedule : public ParameterSchedule {
protected:
    double _finalRate;
public:
    LinearParameterSchedule(double base, double tgtFactor);
    virtual double getValue(double progress);
};

class ExpParameterSchedule : public ParameterSchedule {
protected:
    double _powBase;
public:
    ExpParameterSchedule(double baseRate, double tgtFactor);
    virtual double getValue(double progress);
};

class DiscreteExpParameterSchedule : public ParameterSchedule {
protected:
    std::vector<double> _rates;
public:
    DiscreteExpParameterSchedule(double baseRate, double tgtFactor, int numSteps);
    virtual double getValue(double progress);
};


#endif    /* LR_CUH */
