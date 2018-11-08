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

#include <string>
#include "../include/lr.cuh"
#include "../include/util.cuh"

/*
 * ==================================
 * ParameterSchedule
 * ==================================
 */
ParameterSchedule& ParameterSchedule::make(PyObject* schedDict) {
    std::string type = pyDictGetString(schedDict, "type");
    PyObject* paramsDict = PyDict_GetItemString(schedDict, "params");
    double base = pyDictGetFloat(paramsDict, "base");
    if (type == "const") {
        return *new ParameterSchedule(base);
    } else {
        double tgtFactor = pyDictGetFloat(paramsDict, "tgtFactor");
        if (type == "linear") {
            return *new LinearParameterSchedule(base, tgtFactor);
        } else if (type == "exp") {
            return *new ExpParameterSchedule(base, tgtFactor);
        } else if (type == "dexp") {
            double numSteps = pyDictGetInt(paramsDict, "numSteps");
            return *new DiscreteExpParameterSchedule(base, tgtFactor, numSteps);
        }
    }
    throw std::string("Unknown learning rate schedule type ") + type;
}

ParameterSchedule::ParameterSchedule(double baseRate)
    : _baseRate(baseRate) {
}

double ParameterSchedule::getValue(double progress) {
    return _baseRate;
}

double ParameterSchedule::getBaseValue() const {
    return _baseRate;
}

ParameterSchedule::~ParameterSchedule() {
}

/*
 * ==================================
 * LinearParameterSchedule
 * ==================================
 */
LinearParameterSchedule::LinearParameterSchedule(double baseRate, double tgtFactor)
: ParameterSchedule(baseRate) {
    _finalRate = baseRate / tgtFactor;
}

double LinearParameterSchedule::getValue(double progress) {
    return _baseRate * (1 - progress) + _finalRate * progress;
}

/*
 * ==================================
 * ExpParameterSchedule
 * ==================================
 */
ExpParameterSchedule::ExpParameterSchedule(double baseRate, double tgtFactor)
: ParameterSchedule(baseRate) {
    _powBase = 1.0 / tgtFactor;
}

double ExpParameterSchedule::getValue(double progress) {
    return _baseRate * std::pow(_powBase, progress);
}

/*
 * ==================================
 * DiscreteExpParameterSchedule
 * ==================================
 */
DiscreteExpParameterSchedule::DiscreteExpParameterSchedule(double baseRate, double tgtFactor, int numSteps)
: ParameterSchedule(baseRate) {
    ExpParameterSchedule elrs(baseRate, tgtFactor);
    double finalRate = baseRate / tgtFactor;
    for (int i = 0; i < numSteps - 1; i++) {
        double progress = double(i) / (numSteps - 1);
        _rates.push_back(elrs.getValue(progress));
    }
    _rates.push_back(finalRate);
    //printf("initialized base %e, final %e, stpes %d\n", baseRate, finalRate, numSteps);
}

double DiscreteExpParameterSchedule::getValue(double progress) {
    for (int i = 0; i < _rates.size(); ++i) {
        if (progress <= double(i + 1) / _rates.size()) {
            return _rates[i];
        }
    }
    return _rates.back();
}

