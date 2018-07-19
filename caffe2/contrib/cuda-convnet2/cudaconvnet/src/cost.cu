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

#include <iostream>
#include "../include/cost.cuh"

using namespace std;

/* 
 * =====================
 * Cost
 * =====================
 */

Cost::Cost() {
}

Cost::Cost(vector<CostLayer*>& costs) {
    for (vector<CostLayer*>::iterator it = costs.begin(); it != costs.end(); ++it) {
        _costMap[(*it)->getName()] = &(*it)->getCost();
        _costCoeffMap[(*it)->getName()] = (*it)->getCoeff();
        _numCases[(*it)->getName()] = (*it)->getNumCases();
    }
}

int Cost::getNumCases() {
    return _numCases.size() == 0 ? 0 : _numCases.begin()->second;
}

map<std::string,int>& Cost::getNumCasesMap() {
    return _numCases;
}

doublev& Cost::operator [](const std::string s) {
    return *_costMap[s];
}

CostMap& Cost::getCostMap() {
    return _costMap;
}

CostCoeffMap& Cost::getCostCoeffMap() {
    return _costCoeffMap;
}

double Cost::getValue() {
    double val = 0;
    for (CostMap::iterator it = _costMap.begin(); it != _costMap.end(); ++it) {
        val += _costCoeffMap[it->first] * (it->second->size() == 0 ? 0 : it->second->at(0));
    }
    return val;
}

Cost& Cost::operator += (Cost& er) {
    CostMap& otherMap = er.getCostMap();
    CostCoeffMap& otherCoeffMap = er.getCostCoeffMap();

    for (CostMap::const_iterator it = otherMap.begin(); it != otherMap.end(); ++it) {
        bool newCost = _costMap.count(it->first) == 0;
        if (newCost) {
            _costMap[it->first] = new doublev();
            _costCoeffMap[it->first] = otherCoeffMap[it->first];
            _numCases[it->first] = er.getNumCasesMap()[it->first];
        } else {
            _numCases[it->first] += er.getNumCasesMap()[it->first];
        }
        
        doublev& myVec = *_costMap[it->first];
        doublev& otherVec = *otherMap[it->first];
        assert(myVec.size() == 0 || otherVec.size() == 0 || myVec.size() == otherVec.size());
        // Add costs from otherVec to me
        for (int i = 0; i < otherVec.size(); i++) {
            if (myVec.size() <= i) {
                myVec.push_back(0);
            }
            myVec[i] += otherVec[i];
        }
    }
    return *this;
}

Cost::~Cost() {
    for (CostMap::const_iterator it = _costMap.begin(); it != _costMap.end(); ++it) {
        delete it->second;
    }
}

void Cost::print() {
    for (CostMap::const_iterator it = _costMap.begin(); it != _costMap.end(); ++it) {
        printf("%s (%.3f): ", it->first.c_str(), _costCoeffMap[it->first]);
        doublev& vec = *_costMap[it->first];
        for (int z = 0; z < vec.size(); ++z) {
            printf("%.3f", vec[z]);
            if (z < vec.size() - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
}
