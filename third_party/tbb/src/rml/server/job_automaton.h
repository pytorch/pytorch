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

#ifndef __RML_job_automaton_H
#define __RML_job_automaton_H

#include "rml_base.h"
#include "tbb/atomic.h"

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings 
    #pragma warning (push)
    #pragma warning (disable: 4244)
#endif

namespace rml {

namespace internal {

//! Finite state machine.   
/**   /--------------\
     /                V
    0 --> 1--> ptr --> -1 
                ^
                |
                |
                V
              ptr|1

"owner" = corresponding server_thread.
Odd states (except -1) indicate that someone is executing code on the job.
Most transitions driven only by owner.
Transition 0-->-1 is driven by non-owner.
Transition ptr->-1 is driven  by owner or non-owner.
*/ 
class job_automaton: no_copy {
private:
    tbb::atomic<intptr_t> my_job;
public:
    /** Created by non-owner */
    job_automaton() {
        my_job = 0;
    }
 
    ~job_automaton() {
        __TBB_ASSERT( my_job==-1, "must plug before destroying" );
    }

    //! Try to transition 0-->1 or ptr-->ptr|1.
    /** Should only be called by owner. */
    bool try_acquire() {
        intptr_t snapshot = my_job;
        if( snapshot==-1 ) {
            return false;
        } else {
            __TBB_ASSERT( (snapshot&1)==0, "already marked that way" );
            intptr_t old = my_job.compare_and_swap( snapshot|1, snapshot );
            __TBB_ASSERT( old==snapshot || old==-1, "unexpected interference" );  
            return old==snapshot;
        }
    }
    //! Transition ptr|1-->ptr
    /** Should only be called by owner. */
    void release() {
        intptr_t snapshot = my_job;
        __TBB_ASSERT( snapshot&1, NULL );
        // Atomic store suffices here.
        my_job = snapshot&~1;
    }

    //! Transition 1-->ptr
    /** Should only be called by owner. */
    void set_and_release( rml::job* job ) {
        intptr_t value = reinterpret_cast<intptr_t>(job);
        __TBB_ASSERT( (value&1)==0, "job misaligned" );
        __TBB_ASSERT( value!=0, "null job" );
        __TBB_ASSERT( my_job==1, "already set, or not marked busy?" );
        // Atomic store suffices here.
        my_job = value;
    }

    //! Transition 0-->-1
    /** If successful, return true. called by non-owner (for TBB and the likes) */
    bool try_plug_null() {
        return my_job.compare_and_swap( -1, 0 )==0;
    }

    //! Try to transition to -1.  If successful, set j to contents and return true.
    /** Called by owner or non-owner. (for OpenMP and the likes) */
    bool try_plug( rml::job*&j ) {
        for(;;) {
            intptr_t snapshot = my_job;
            if( snapshot&1 ) {
                j = NULL;
                return false;
            } 
            // Not busy
            if( my_job.compare_and_swap( -1, snapshot )==snapshot ) {
                j = reinterpret_cast<rml::job*>(snapshot);
                return true;
            } 
            // Need to retry, because current thread may be non-owner that read a 0, and owner might have
            // caused transition 0->1->ptr after we took our snapshot.
        }
    }

    /** Called by non-owner to wait for transition to ptr. */
    rml::job* wait_for_job() const {
        intptr_t snapshot;
        for(;;) {
            snapshot = my_job;
            if( snapshot&~1 ) break;
            __TBB_Yield();
        }
        __TBB_ASSERT( snapshot!=-1, "wait on plugged job_automaton" );
        return reinterpret_cast<rml::job*>(snapshot&~1);
    }
};

} // namespace internal
} // namespace rml


#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warning 4244 are back

#endif /* __RML_job_automaton_H */
