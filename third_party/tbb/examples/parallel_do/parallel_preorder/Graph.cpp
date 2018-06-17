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

#include <cstdlib>
#include "Graph.h"
#include <iostream>

using namespace std;

void Graph::create_random_dag( size_t number_of_nodes ) {
    my_vertex_set.resize(number_of_nodes);
    for( size_t k=0; k<number_of_nodes; ++k ) {
        Cell& c = my_vertex_set[k];
        int op = int((rand()>>8)%5u);
        if( op>int(k) ) op = int(k);
        switch( op ) {
            default:
                c.op = OP_VALUE;
                c.value = Cell::value_type((float)k);
                break;
            case 1:
                c.op = OP_NEGATE;
                break;
            case 2:
                c.op = OP_SUB;
                break;
            case 3: 
                c.op = OP_ADD;
                break;
            case 4: 
                c.op = OP_MUL;
                break;
        }
        for( int j=0; j<ArityOfOp[c.op]; ++j ) {
            Cell& input = my_vertex_set[rand()%k];
            c.input[j] = &input;
        }
    }
}

void Graph::print() {
    for( size_t k=0; k<my_vertex_set.size(); ++k ) {
        std::cout<<"Cell "<<k<<":";
        for( size_t j=0; j<my_vertex_set[k].successor.size(); ++j )
            std::cout<<" "<<int(my_vertex_set[k].successor[j] - &my_vertex_set[0]);
        std::cout<<std::endl;
    }
}

void Graph::get_root_set( vector<Cell*>& root_set ) {
    for( size_t k=0; k<my_vertex_set.size(); ++k ) {
        my_vertex_set[k].successor.clear();
    }
    root_set.clear();
    for( size_t k=0; k<my_vertex_set.size(); ++k ) {
        Cell& c = my_vertex_set[k];
        c.ref_count = ArityOfOp[c.op];
        for( int j=0; j<ArityOfOp[c.op]; ++j ) {
            c.input[j]->successor.push_back(&c);
        }
        if( ArityOfOp[c.op]==0 )
            root_set.push_back(&my_vertex_set[k]);
    }
}

void Cell::update() {
    switch( op ) {
        case OP_VALUE:
            break;
        case OP_NEGATE:
            value = -(input[0]->value);
            break;
        case OP_ADD:
            value = input[0]->value + input[1]->value;
            break;
        case OP_SUB:
            value = input[0]->value - input[1]->value;
            break;
        case OP_MUL:
            value = input[0]->value * input[1]->value;
            break;
    }
}

