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
// Self-organizing map in TBB flow::graph
//
// we will do a color map (the simple example.)
//
//  serial algorithm
//
//       initialize map with vectors (could be random, gradient, or something else)
//       for some number of iterations
//           update radius r, weight of change L 
//           for each example V
//               find the best matching unit
//               for each part of map within radius of BMU W
//                   update vector:  W(t+1) = W(t) + w(dist)*L*(V - W(t))

#include "som.h"
#include "tbb/task.h"

std::ostream& operator<<( std::ostream &out, const SOM_element &s) {
    out << "(";
    for(int i=0;i<(int)s.w.size();++i) {
        out << s.w[i];
        if(i < (int)s.w.size()-1) {
            out << ",";
        }
    }
    out << ")";
    return out;
}

void remark_SOM_element(const SOM_element &s) {
    printf("(");
    for(int i=0;i<(int)s.w.size();++i) {
        printf("%g",s.w[i]);
        if(i < (int)s.w.size()-1) {
            printf(",");
        }
    }
    printf(")");
}

std::ostream& operator<<( std::ostream &out, const search_result_type &s) {
    out << "<";
    out << get<RADIUS>(s);
    out <<  ", " << get<XV>(s);
    out << ", ";
    out << get<YV>(s);
    out << ">";
    return out;
}

void remark_search_result_type(const search_result_type &s) {
    printf("<%g,%d,%d>", get<RADIUS>(s), get<XV>(s), get<YV>(s));
}

double
randval( double lowlimit, double highlimit) {
    return double(rand()) / double(RAND_MAX) * (highlimit - lowlimit) + lowlimit;
}

void
find_data_ranges(teaching_vector_type &teaching, SOM_element &max_range, SOM_element &min_range ) {
    if(teaching.size() == 0) return;
    max_range = min_range = teaching[0];
    for(int i = 1; i < (int)teaching.size(); ++i) {
        max_range.elementwise_max(teaching[i]);
        min_range.elementwise_min(teaching[i]);
    }
} 

void add_fraction_of_difference( SOM_element &to, SOM_element const &from, double frac) {
    for(int i = 0; i < (int)from.size(); ++i) {
        to[i] += frac*(from[i] - to[i]);
    }
}

double
distance_squared(SOM_element x, SOM_element y) {
    double rval = 0.0; for(int i=0;i<(int)x.size();++i) {
        double diff = x[i] - y[i];
        rval += diff*diff;
    }
    return rval;
}

void SOMap::initialize(InitializeType it, SOM_element &max_range, SOM_element &min_range) {
    for(int x = 0; x < xMax; ++x) {
        for(int y = 0; y < yMax; ++y) {
            for( int i = 0; i < (int)max_range.size(); ++i) {
                if(it == InitializeRandom) {
                    my_map[x][y][i] = (randval(min_range[i], max_range[i]));
                }
                else if(it == InitializeGradient) {
                    my_map[x][y][i] = ((double)(x+y)/(xMax+yMax)*(max_range[i]-min_range[i]) + min_range[i]);
                }
            }
        }
    }
}

// subsquare [low,high)
double
SOMap::BMU_range( const SOM_element &s, int &xval, int &yval, subsquare_type &r) {
    double min_distance_squared = DBL_MAX;
    task &my_task = task::self();
    int min_x = -1;
    int min_y = -1;
    for(int x = r.rows().begin(); x != r.rows().end(); ++x) {
        for( int y = r.cols().begin(); y != r.cols().end(); ++y) {
            double dist = distance_squared(s,my_map[x][y]);
            if(dist < min_distance_squared) {
                min_distance_squared = dist;
                min_x = x;
                min_y = y;
            }
            if(cancel_test && my_task.is_cancelled()) {
                xval = r.rows().begin();
                yval = r.cols().begin();
                return DBL_MAX;
            }
        }
    }
    xval = min_x;
    yval = min_y;
    return sqrt(min_distance_squared);
}

void
SOMap::epoch_update_range( SOM_element const &s, int epoch, int min_x, int min_y, double radius, double learning_rate, blocked_range<int> &r) {
    int min_xiter = (int)((double)min_x - radius);
    if(min_xiter < 0) min_xiter = 0;
    int max_xiter = (int)((double)min_x + radius);
    if(max_xiter > (int)my_map.size()-1) max_xiter = (int)my_map.size()-1;
    for(int xx = r.begin(); xx <= r.end(); ++xx) {
        double xrsq = (xx-min_x)*(xx-min_x);
        double ysq = radius*radius - xrsq;  // max extent of y influence
        double yd;
        if(ysq > 0) {
            yd = sqrt(ysq);
            int lb = (int)(min_y - yd);
            int ub = (int)(min_y + yd);
            for(int yy = lb; yy < ub; ++yy) {
                if(yy >= 0 && yy < (int)my_map[xx].size()) {
                    // [xx, yy] is in the range of the update.
                    double my_rsq = xrsq + (yy-min_y)*(yy-min_y);  // distance from BMU squared
                    double theta = exp(-(radius*radius) /(2.0* my_rsq)); 
                    add_fraction_of_difference(my_map[xx][yy], s, theta * learning_rate);
                }
            }
        }
    }
}

void SOMap::teach(teaching_vector_type &in) {
    for(int i = 0; i < nPasses; ++i ) {
        int j = (int)(randval(0, (double)in.size()));  // this won't be reproducible.
        if(j == in.size()) --j;
        
        int min_x = -1;
        int min_y = -1;
        subsquare_type br2(0, (int)my_map.size(), 1, 0, (int)my_map[0].size(), 1);
        (void) BMU_range(in[j],min_x, min_y, br2);  // just need min_x, min_y
        // radius of interest
        double radius = max_radius * exp(-(double)i*radius_decay_rate);
        // update circle is min_xiter to max_xiter inclusive.
        double learning_rate = max_learning_rate * exp( -(double)i * learning_decay_rate);
        epoch_update(in[j], i, min_x, min_y, radius, learning_rate);
    }
}

void SOMap::debug_output() {
    printf("SOMap:\n");
    for(int i = 0; i < (int)(this->my_map.size()); ++i) {
        for(int j = 0; j < (int)(this->my_map[i].size()); ++j) {
            printf( "map[%d, %d] == ", i, j );
            remark_SOM_element( this->my_map[i][j] );
            printf("\n");
        }
    }
}

#define RED 0
#define GREEN 1
#define BLUE 2

void readInputData() {
    my_teaching.push_back(SOM_element());
    my_teaching.push_back(SOM_element());
    my_teaching.push_back(SOM_element());
    my_teaching.push_back(SOM_element());
    my_teaching.push_back(SOM_element());
    my_teaching[0][RED] = 1.0; my_teaching[0][GREEN] = 0.0; my_teaching[0][BLUE] = 0.0;
    my_teaching[1][RED] = 0.0; my_teaching[1][GREEN] = 1.0; my_teaching[1][BLUE] = 0.0;
    my_teaching[2][RED] = 0.0; my_teaching[2][GREEN] = 0.0; my_teaching[2][BLUE] = 1.0;
    my_teaching[3][RED] = 0.3; my_teaching[3][GREEN] = 0.3; my_teaching[3][BLUE] = 0.0;
    my_teaching[4][RED] = 0.5; my_teaching[4][GREEN] = 0.5; my_teaching[4][BLUE] = 0.9;
}
