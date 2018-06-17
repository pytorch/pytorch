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

#ifndef tbb_tests_harness_concurrency_tracker_H
#define tbb_tests_harness_concurrency_tracker_H

#include "harness_assert.h"
#include "harness_barrier.h"
#include "tbb/atomic.h"
#include "../tbb/tls.h"
// Note: This file is used by RML tests which do not link TBB.
// Functionality that requires TBB binaries must be guarded by !__TBB_NO_IMPLICIT_LINKAGE
#if !defined(__TBB_NO_IMPLICIT_LINKAGE)
#include "tbb/mutex.h"
#include "tbb/task.h"
#include "tbb/combinable.h"
#include "tbb/parallel_for.h"
#include <functional> // for std::plus
#include "harness.h" // for Harness::NoCopy
#endif

namespace Harness {

static tbb::atomic<unsigned> ctInstantParallelism;
static tbb::atomic<unsigned> ctPeakParallelism;
static tbb::internal::tls<uintptr_t>  ctNested;

class ConcurrencyTracker {
    bool    m_Outer;

    static void Started () {
        unsigned p = ++ctInstantParallelism;
        unsigned q = ctPeakParallelism;
        while( q<p ) {
            q = ctPeakParallelism.compare_and_swap(p,q);
        }
    }

    static void Stopped () {
        ASSERT ( ctInstantParallelism > 0, "Mismatched call to ConcurrencyTracker::Stopped()" );
        --ctInstantParallelism;
    }
public:
    ConcurrencyTracker() : m_Outer(false) {
        uintptr_t nested = ctNested;
        ASSERT (nested == 0 || nested == 1, NULL);
        if ( !ctNested ) {
            Started();
            m_Outer = true;
            ctNested = 1;
        }
    }
    ~ConcurrencyTracker() {
        if ( m_Outer ) {
            Stopped();
            ctNested = 0;
        }
    }

    static unsigned PeakParallelism() { return ctPeakParallelism; }
    static unsigned InstantParallelism() { return ctInstantParallelism; }

    static void Reset() {
        ASSERT (ctInstantParallelism == 0, "Reset cannot be called when concurrency tracking is underway");
        ctInstantParallelism = ctPeakParallelism = 0;
    }
}; // ConcurrencyTracker

#if !defined(__TBB_NO_IMPLICIT_LINKAGE)
struct ExactConcurrencyLevel : NoCopy {
    typedef tbb::combinable<size_t> Combinable;
private:
    Harness::SpinBarrier       *myBarrier;
    // count unique worker threads
    Combinable                 *myUniqueThreads;
    mutable tbb::atomic<size_t> myActiveBodyCnt;
    // output parameter for parallel_for body to report that max is reached
    mutable bool                myReachedMax;
    // zero timeout means no barrier is used during concurrency level detection
    const double                myTimeout;
    const size_t                myConcLevel;
    const bool                  myCrashOnFail;

    static tbb::mutex global_mutex;

    ExactConcurrencyLevel(double timeout, size_t concLevel, Combinable *uniq, bool crashOnFail) :
        myBarrier(NULL), myUniqueThreads(uniq), myReachedMax(false),
        myTimeout(timeout), myConcLevel(concLevel), myCrashOnFail(crashOnFail) {
        myActiveBodyCnt = 0;
    }
    bool run() {
        const int LOOP_ITERS = 100;
        tbb::combinable<size_t> uniq;
        Harness::SpinBarrier barrier((unsigned)myConcLevel, /*throwaway=*/true);
        if (myTimeout != 0.)
            myBarrier = &barrier;
        if (!myUniqueThreads)
            myUniqueThreads = &uniq;
        tbb::parallel_for((size_t)0, myConcLevel*LOOP_ITERS, *this, tbb::simple_partitioner());
        return myReachedMax;
    }
public:
    void operator()(size_t) const {
        size_t v = ++myActiveBodyCnt;
        ASSERT(v <= myConcLevel, "Number of active bodies is too high.");
        if (v == myConcLevel) // record that the max expected concurrency was observed
            myReachedMax = true;
        // try to get barrier when 1st time in the thread
        if (myBarrier && !myBarrier->timed_wait_noerror(myTimeout))
            ASSERT(!myCrashOnFail, "Timeout was detected.");

        myUniqueThreads->local() = 1;
        for (int i=0; i<100; i++)
            __TBB_Pause(1);
        --myActiveBodyCnt;
    }

    enum Mode {
        None,
        // When multiple blocking checks are performed, there might be not enough
        // concurrency for all of them. Serialize check() calls.
        Serialize
    };

    // check that we have never got more than concLevel threads,
    // and that in some moment we saw exactly concLevel threads
    static void check(size_t concLevel, Mode m = None) {
        ExactConcurrencyLevel o(30., concLevel, NULL, /*crashOnFail=*/true);

        tbb::mutex::scoped_lock lock;
        if (m == Serialize)
            lock.acquire(global_mutex);
        bool ok = o.run();
        ASSERT(ok, NULL);
    }

    static bool isEqual(size_t concLevel) {
        ExactConcurrencyLevel o(3., concLevel, NULL, /*crashOnFail=*/false);
        return o.run();
    }

    static void checkLessOrEqual(size_t concLevel, tbb::combinable<size_t> *unique) {
        ExactConcurrencyLevel o(0., concLevel, unique, /*crashOnFail=*/true);

        o.run(); // ignore result, as without a barrier it is not reliable
        const size_t num = unique->combine(std::plus<size_t>());
        ASSERT(num<=concLevel, "Too many workers observed.");
    }
};

tbb::mutex ExactConcurrencyLevel::global_mutex;

#endif /* !defined(__TBB_NO_IMPLICIT_LINKAGE) */

} // namespace Harness

#endif /* tbb_tests_harness_concurrency_tracker_H */
