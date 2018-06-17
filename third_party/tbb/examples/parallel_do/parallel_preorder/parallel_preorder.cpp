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

#include "tbb/parallel_do.h"
#include <vector>
#include <algorithm>
#include "Graph.h"


class Body {
public:
    Body() {};

    //------------------------------------------------------------------------
    // Following signatures are required by parallel_do
    //------------------------------------------------------------------------
    typedef Cell* argument_type;

    void operator()( Cell* c, tbb::parallel_do_feeder<Cell*>& feeder ) const {
        c->update();
        // Restore ref_count in preparation for subsequent traversal.
        c->ref_count = ArityOfOp[c->op];
        for( size_t k=0; k<c->successor.size(); ++k ) {
            Cell* successor = c->successor[k];
            // ref_count is used for inter-task synchronization.
            // Correctness checking tools might not take this into account, and report
            // data races between different tasks, that are actually synchronized.
            if( 0 == --(successor->ref_count) ) {
                feeder.add( successor );
            }
        }
    }
};

void ParallelPreorderTraversal( const std::vector<Cell*>& root_set ) {
    tbb::parallel_do(root_set.begin(), root_set.end(),Body());
}


