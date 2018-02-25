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

#ifndef COST_CUH
#define	COST_CUH

#include <vector>
#include <map>
#include <helper_cuda.h>

#include "layer.cuh"
#include "util.cuh"

class CostLayer;

/*
 * Wrapper for dictionary mapping cost name to vector of returned values.
 */
class Cost {
protected:
    std::map<std::string,int> _numCases;
    CostMap _costMap;
    CostCoeffMap _costCoeffMap;
    std::map<std::string,int>& getNumCasesMap();
public:
    Cost();
    Cost(std::vector<CostLayer*>& costs);
    doublev& operator [](const std::string s);
    CostMap& getCostMap();
    CostCoeffMap& getCostCoeffMap();
    int getNumCases();
    /*
     * Returns sum of first values returned by all the CostLayers, weighted by the cost coefficients.
     */
    double getValue();
    Cost& operator += (Cost& er);
    virtual ~Cost();
    void print();
};


#endif	/* COST_CUH */

