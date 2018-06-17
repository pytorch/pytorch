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

#include "tbb/task.h"
#include "harness.h"

//! Helper for verifying that old use cases of spawn syntax still work.
tbb::task* GetTaskPtr( int& counter ) {
    ++counter;
    return NULL;
}

class TaskGenerator: public tbb::task {
    int m_ChildCount;
    int m_Depth;

public:
    TaskGenerator( int child_count, int _depth ) : m_ChildCount(child_count), m_Depth(_depth) {}
    ~TaskGenerator( ) { m_ChildCount = m_Depth = -125; }

    tbb::task* execute() __TBB_override {
        ASSERT( m_ChildCount>=0 && m_Depth>=0, NULL );
        if( m_Depth>0 ) {
            recycle_as_safe_continuation();
            set_ref_count( m_ChildCount+1 );
            int k=0;
            for( int j=0; j<m_ChildCount; ++j ) {
                tbb::task& t = *new( allocate_child() ) TaskGenerator(m_ChildCount/2,m_Depth-1);
                GetTaskPtr(k)->spawn(t);
            }
            ASSERT(k==m_ChildCount,NULL);
            --m_Depth;
            __TBB_Yield();
            ASSERT( state()==recycle && ref_count()>0, NULL);
        }
        return NULL;
    }
};
