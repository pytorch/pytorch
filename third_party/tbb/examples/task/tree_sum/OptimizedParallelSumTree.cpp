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

#include "common.h"
#include "tbb/task.h"

class OptimizedSumTask: public tbb::task {
    Value* const sum;
    TreeNode* root;
    bool is_continuation;
    Value x, y;
public:
    OptimizedSumTask( TreeNode* root_, Value* sum_ ) : root(root_), sum(sum_), is_continuation(false) {
    }
    tbb::task* execute() /*override*/ {
        tbb::task* next = NULL;
        if( !is_continuation ) {
            if( root->node_count<1000 ) {
                *sum = SerialSumTree(root);
            } else {
                // Create tasks before spawning any of them.
                tbb::task* a = NULL;
                tbb::task* b = NULL;
                if( root->left )
                    a = new( allocate_child() ) OptimizedSumTask(root->left,&x);
                if( root->right )
                    b = new( allocate_child() ) OptimizedSumTask(root->right,&y);
                recycle_as_continuation();
                is_continuation = true;
                set_ref_count( (a!=NULL)+(b!=NULL) );
                if( a ) {
                    if( b ) spawn(*b);
                } else 
                    a = b;
                next = a;
            }
        } else {
            *sum = root->value;
            if( root->left ) *sum += x;
            if( root->right ) *sum += y;
        } 
        return next;
    }
};

Value OptimizedParallelSumTree( TreeNode* root ) {
    Value sum;
    OptimizedSumTask& a = *new(tbb::task::allocate_root()) OptimizedSumTask(root,&sum);
    tbb::task::spawn_root_and_wait(a);
    return sum;
}

