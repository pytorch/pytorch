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

#ifndef __RML_wait_counter_H
#define __RML_wait_counter_H

#include "thread_monitor.h"
#include "tbb/atomic.h"

namespace rml {
namespace internal {

class wait_counter {
    thread_monitor my_monitor;
    tbb::atomic<int> my_count;
    tbb::atomic<int> n_transients;
public:
    wait_counter() { 
        // The "1" here is subtracted by the call to "wait".
        my_count=1;
        n_transients=0;
    }

    //! Wait for number of operator-- invocations to match number of operator++ invocations.
    /** Exactly one thread should call this method. */
    void wait() {
        int k = --my_count;
        __TBB_ASSERT( k>=0, "counter underflow" );
        if( k>0 ) {
            thread_monitor::cookie c;
            my_monitor.prepare_wait(c);
            if( my_count )
                my_monitor.commit_wait(c);
            else 
                my_monitor.cancel_wait();
        }
        while( n_transients>0 )
            __TBB_Yield();
    }
    void operator++() {
        ++my_count;
    }
    void operator--() {
        ++n_transients;
        int k = --my_count;
        __TBB_ASSERT( k>=0, "counter underflow" );
        if( k==0 ) 
            my_monitor.notify();
        --n_transients;
    }
};

} // namespace internal
} // namespace rml

#endif /* __RML_wait_counter_H */
