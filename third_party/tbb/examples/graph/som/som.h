/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

//
// Self-organizing map
//
// support for self-ordering maps
#ifndef __SOM_H__
#define __SOM_H__

#include <vector>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstdio>

#include "tbb/flow_graph.h"
#include "tbb/blocked_range2d.h"

using namespace tbb;
using namespace tbb::flow;

typedef blocked_range2d<int> subsquare_type;
typedef tuple<double,int,int> search_result_type;

std::ostream& operator<<( std::ostream &out, const search_result_type &s);

#define RADIUS 0  // for the std::gets
#define XV     1
#define YV     2

// to have single definitions of static variables, define _MAIN_C_ in the main program
// 
#ifdef _MAIN_C_
#define DEFINE // nothing
#define INIT(n) = n
#else // not in main file
#define DEFINE extern
#define INIT(n) // nothing
#endif  // _MAIN_C_

DEFINE int nElements INIT(3);  // length of input vectors, matching vector in map
DEFINE double max_learning_rate INIT(0.8);  // decays exponentially
DEFINE double radius_decay_rate;
DEFINE double learning_decay_rate INIT(0.005);
DEFINE double max_radius;
DEFINE bool extra_debug INIT(false);
DEFINE bool cancel_test INIT(false);

DEFINE int xMax INIT(100);
DEFINE int yMax INIT(100);
DEFINE int nPasses INIT(100);

enum InitializeType { InitializeRandom, InitializeGradient };
#define RED 0
#define GREEN 1
#define BLUE 2
class SOM_element;
void remark_SOM_element(const SOM_element &s);

// all SOM_element vectors are the same length (nElements), so we do not have
// to range-check the vector accesses.
class SOM_element {
    std::vector<double> w;
public:
    friend std::ostream& operator<<( std::ostream &out, const SOM_element &s);
    friend void remark_SOM_element(const SOM_element &s);
    SOM_element() : w(nElements,0.0) {}
    double &operator[](int indx) { return w.at(indx); }
    const double &operator[](int indx) const { return w.at(indx); }
    bool operator==(SOM_element const &other) const {
        for(size_t i=0;i<size();++i) {
            if(w[i] != other.w[i]) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(SOM_element const &other) const { return !operator==(other); }
    void elementwise_max(SOM_element const &other) {
        for(size_t i = 0; i < w.size(); ++i) if(w[i] < other.w[i]) w[i] = other.w[i];
    }
    void elementwise_min(SOM_element const &other) {
        for(size_t i = 0; i < w.size(); ++i) if(w[i] > other.w[i]) w[i] = other.w[i];
    }
    size_t size() const { return w.size(); }
};

typedef std::vector<SOM_element> teaching_vector_type;

DEFINE SOM_element max_range;
DEFINE SOM_element min_range;

extern double randval( double lowlimit, double highlimit);

extern void find_data_ranges(teaching_vector_type &teaching, SOM_element &max_range, SOM_element &min_range );

extern void add_fraction_of_difference( SOM_element &to, SOM_element &from, double frac);

DEFINE teaching_vector_type my_teaching;

class SOMap {
    std::vector< std::vector< SOM_element > > my_map;
public:
    SOMap(int xSize, int ySize) {
        my_map.reserve(xSize);
        for(int i = 0; i < xSize; ++i) {
            my_map.push_back(teaching_vector_type());
            my_map[i].reserve(ySize);
            for(int j = 0; j < ySize;++j) {
                my_map[i].push_back(SOM_element());
            }
        }
    }
    size_t size() { return my_map.size(); }
    void initialize(InitializeType it, SOM_element &max_range, SOM_element &min_range);
    teaching_vector_type &operator[](int indx) { return my_map[indx]; }
    SOM_element &at(int xVal, int yVal) { return my_map[xVal][yVal]; }
    SOM_element &at(search_result_type const &s) { return my_map[flow::get<1>(s)][flow::get<2>(s)]; }
    void epoch_update( SOM_element const &s, int epoch, int min_x, int min_y, double radius, double learning_rate) {
        int min_xiter = (int)((double)min_x - radius);
        if(min_xiter < 0) min_xiter = 0;
        int max_xiter = (int)((double)min_x + radius);
        if(max_xiter > (int)my_map.size()-1) max_xiter = (int)(my_map.size()-1);
        blocked_range<int> br1(min_xiter, max_xiter, 1);
        epoch_update_range(s, epoch, min_x, min_y, radius, learning_rate, br1);
    }
    void epoch_update_range( SOM_element const &s, int epoch, int min_x, int min_y, double radius, double learning_rate, blocked_range<int> &r);
    void teach( teaching_vector_type &id);
    void debug_output();
    // find BMU given an input, returns distance
    double BMU_range(const SOM_element &s, int &xval, int &yval, subsquare_type &r);
    double BMU(const SOM_element &s, int &xval, int &yval) {
        subsquare_type br(0,(int)my_map.size(),1,0,(int)my_map[0].size(),1);
        return BMU_range(s, xval, yval, br);
    }
};

extern double distance_squared(SOM_element x, SOM_element y);
void remark_SOM_element(const SOM_element &s);

extern void readInputData();
#endif // __SOM_H__
