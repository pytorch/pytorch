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

#include "tbb/atomic.h"
#include "tbb/tick_count.h"

#ifndef harness_barrier_H
#define harness_barrier_H

namespace Harness {

//! Spin WHILE the value of the variable is equal to a given value
/** T and U should be comparable types. */
class TimedWaitWhileEq {
    //! Assignment not allowed
    void operator=( const TimedWaitWhileEq& );
    double &my_limit;
public:
    TimedWaitWhileEq(double &n_seconds) : my_limit(n_seconds) {}
    TimedWaitWhileEq(const TimedWaitWhileEq &src) : my_limit(src.my_limit) {}
    template<typename T, typename U>
    void operator()( const volatile T& location, U value ) const {
        tbb::tick_count start = tbb::tick_count::now();
        double time_passed;
        do {
            time_passed = (tbb::tick_count::now()-start).seconds();
            if( time_passed < 0.0001 ) __TBB_Pause(10); else __TBB_Yield();
        } while( time_passed < my_limit && location == value);
        my_limit -= time_passed;
    }
};
//! Spin WHILE the value of the variable is equal to a given value
/** T and U should be comparable types. */
class WaitWhileEq {
    //! Assignment not allowed
    void operator=( const WaitWhileEq& );
public:
    template<typename T, typename U>
    void operator()( const volatile T& location, U value ) const {
        tbb::internal::spin_wait_while_eq(location, value);
    }
};
class SpinBarrier
{
    unsigned numThreads;
    tbb::atomic<unsigned> numThreadsFinished; // reached the barrier in this epoch
    // the number of times the barrier was opened; TODO: move to a separate cache line
    tbb::atomic<unsigned> epoch;
    // a throwaway barrier can be used only once, then wait() becomes a no-op
    bool throwaway;

    struct DummyCallback {
        void operator() () const {}
        template<typename T, typename U>
        void operator()( const T&, U) const {}
    };

    SpinBarrier( const SpinBarrier& );    // no copy ctor
    void operator=( const SpinBarrier& ); // no assignment
public:
    SpinBarrier( unsigned nthreads = 0, bool throwaway_ = false ) {
        initialize(nthreads, throwaway_);
    }
    void initialize( unsigned nthreads, bool throwaway_ = false ) {
        numThreads = nthreads;
        numThreadsFinished = 0;
        epoch = 0;
        throwaway = throwaway_;
    }

    // Returns whether this thread was the last to reach the barrier.
    // onWaitCallback is called by a thread for waiting;
    // onOpenBarrierCallback is called by the last thread before unblocking other threads.
    template<typename WaitEq, typename Callback>
    bool custom_wait(const WaitEq &onWaitCallback, const Callback &onOpenBarrierCallback)
    {
        if (throwaway && epoch)
            return false;
        unsigned myEpoch = epoch;
        unsigned myNumThreads = numThreads; // read it before the increment
        int threadsLeft = myNumThreads - numThreadsFinished.fetch_and_increment() - 1;
        ASSERT(threadsLeft>=0, "Broken barrier");
        if (threadsLeft > 0) {
            /* this thread is not the last; wait until the epoch changes & return false */
            onWaitCallback(epoch, myEpoch);
            return false;
        }
        /* This thread is the last one at the barrier in this epoch */
        onOpenBarrierCallback();
        /* reset the barrier, increment the epoch, and return true */
        threadsLeft = numThreadsFinished -= myNumThreads;
        ASSERT( threadsLeft == 0, "Broken barrier");
        /* wakes up threads waiting to exit in this epoch */
        myEpoch -= epoch++;
        ASSERT( myEpoch == 0, "Broken barrier");
        return true;
    }
    bool timed_wait_noerror(double n_seconds) {
        custom_wait(TimedWaitWhileEq(n_seconds), DummyCallback());
        return n_seconds >= 0.0001;
    }
    bool timed_wait(double n_seconds, const char *msg="Time is out while waiting on a barrier") {
        bool is_last = custom_wait(TimedWaitWhileEq(n_seconds), DummyCallback());
        ASSERT( n_seconds >= 0, msg); // TODO: refactor to avoid passing msg here and rising assertion
        return is_last;
    }
    // onOpenBarrierCallback is called by the last thread before unblocking other threads.
    template<typename Callback>
    bool wait(const Callback &onOpenBarrierCallback) {
        return custom_wait(WaitWhileEq(), onOpenBarrierCallback);
    }
    bool wait(){
        return wait(DummyCallback());
    }
    //! signal to the barrier, rather a semaphore functionality
    bool signal_nowait() {
        return custom_wait(DummyCallback(),DummyCallback());
    }
};

}

#endif //harness_barrier_H
