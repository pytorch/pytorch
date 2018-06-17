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

#ifndef TREE_MAKER_H_
#define TREE_MAKER_H_

#include "tbb/tick_count.h"
#include "tbb/task.h"

static double Pi = 3.14159265358979;

const bool tbbmalloc = true;
const bool stdmalloc = false;

template<bool use_tbbmalloc>
class TreeMaker {

    class SubTreeCreationTask: public tbb::task {
        TreeNode*& my_root;
        bool is_continuation;
        typedef TreeMaker<use_tbbmalloc> MyTreeMaker;

    public:
        SubTreeCreationTask( TreeNode*& root, long number_of_nodes ) : my_root(root), is_continuation(false) {
            my_root = MyTreeMaker::allocate_node();
            my_root->node_count = number_of_nodes;
            my_root->value = Value(Pi*number_of_nodes);
        }

        tbb::task* execute() /*override*/ {
            tbb::task* next = NULL;
            if( !is_continuation ) {
                long subtree_size = my_root->node_count - 1;
                if( subtree_size<1000 ) { /* grainsize */
                    my_root->left  = MyTreeMaker::do_in_one_thread(subtree_size/2);
                    my_root->right = MyTreeMaker::do_in_one_thread(subtree_size - subtree_size/2);
                } else {
                    // Create tasks before spawning any of them.
                    tbb::task* a = new( allocate_child() ) SubTreeCreationTask(my_root->left,subtree_size/2);
                    tbb::task* b = new( allocate_child() ) SubTreeCreationTask(my_root->right,subtree_size - subtree_size/2);
                    recycle_as_continuation();
                    is_continuation = true;
                    set_ref_count(2);
                    spawn(*b);
                    next = a;
                }
            } 
            return next;
        }
    };

public:
    static TreeNode* allocate_node() {
        return use_tbbmalloc? tbb::scalable_allocator<TreeNode>().allocate(1) : new TreeNode;
    }

    static TreeNode* do_in_one_thread( long number_of_nodes ) {
        if( number_of_nodes==0 ) {
            return NULL;
        } else {
            TreeNode* n = allocate_node();
            n->node_count = number_of_nodes;
            n->value = Value(Pi*number_of_nodes);
            --number_of_nodes;
            n->left  = do_in_one_thread( number_of_nodes/2 ); 
            n->right = do_in_one_thread( number_of_nodes - number_of_nodes/2 );
            return n;
        }
    }

    static TreeNode* do_in_parallel( long number_of_nodes ) {
        TreeNode* root_node;
        SubTreeCreationTask& a = *new(tbb::task::allocate_root()) SubTreeCreationTask(root_node, number_of_nodes);
        tbb::task::spawn_root_and_wait(a);
        return root_node;
    }

    static TreeNode* create_and_time( long number_of_nodes, bool silent=false ) {
        tbb::tick_count t0, t1;
        TreeNode* root = allocate_node();
        root->node_count = number_of_nodes;
        root->value = Value(Pi*number_of_nodes);
        --number_of_nodes;

        t0 = tbb::tick_count::now();
        root->left  = do_in_one_thread( number_of_nodes/2 );
        t1 = tbb::tick_count::now();
        if ( !silent ) printf ("%24s: time = %.1f msec\n", "half created serially", (t1-t0).seconds()*1000);

        t0 = tbb::tick_count::now();
        root->right = do_in_parallel( number_of_nodes - number_of_nodes/2 );
        t1 = tbb::tick_count::now();
        if ( !silent ) printf ("%24s: time = %.1f msec\n", "half done in parallel", (t1-t0).seconds()*1000);

        return root;
    }
};

#endif // TREE_MAKER_H_
