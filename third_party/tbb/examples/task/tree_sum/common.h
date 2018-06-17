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

typedef float Value;

struct TreeNode {
    //! Pointer to left subtree
    TreeNode* left; 
    //! Pointer to right subtree
    TreeNode* right;
    //! Number of nodes in this subtree, including this node.
    long node_count;
    //! Value associated with the node.
    Value value;
};

Value SerialSumTree( TreeNode* root );
Value SimpleParallelSumTree( TreeNode* root );
Value OptimizedParallelSumTree( TreeNode* root );
