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

// test critical section
//
#include "tbb/critical_section.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/tick_count.h"
#include "harness_assert.h"
#include "harness.h"
#include <math.h>

#include "harness_barrier.h"
Harness::SpinBarrier sBarrier;
tbb::critical_section cs;
const int MAX_WORK = 300;

struct BusyBody : NoAssign {
    tbb::enumerable_thread_specific<double> &locals;
    const int nThread;
    const int WorkRatiox100;
    int &unprotected_count;
    bool test_throw;

    BusyBody( int nThread_, int workRatiox100_, tbb::enumerable_thread_specific<double> &locals_, int &unprotected_count_, bool test_throw_) :
        locals(locals_),
        nThread(nThread_),
        WorkRatiox100(workRatiox100_),
        unprotected_count(unprotected_count_),
        test_throw(test_throw_) {
        sBarrier.initialize(nThread_);
    }

    void operator()(const int /* threadID */ ) const {
        int nIters = MAX_WORK/nThread;
        sBarrier.wait();
        tbb::tick_count t0 = tbb::tick_count::now();
        for(int j = 0; j < nIters; j++) {

            for(int i = 0; i < MAX_WORK * (100 - WorkRatiox100); i++) {
                locals.local() += 1.0;
            }
            cs.lock();
            ASSERT( !cs.try_lock(), "recursive try_lock must fail" );
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
            if(test_throw && j == (nIters / 2)) {
                bool was_caught = false,
                     unknown_exception = false;
                try {
                    cs.lock();
                }
                catch(tbb::improper_lock& e) {
                    ASSERT( e.what(), "Error message is absent" );
                    was_caught = true;
                }
                catch(...) {
                    was_caught = unknown_exception = true;
                }
                ASSERT(was_caught, "Recursive lock attempt did not throw");
                ASSERT(!unknown_exception, "tbb::improper_lock exception is expected");
            }
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN  */
            for(int i = 0; i < MAX_WORK * WorkRatiox100; i++) {
                locals.local() += 1.0;
            }
            unprotected_count++;
            cs.unlock();
        }
        locals.local() = (tbb::tick_count::now() - t0).seconds();
    }
};

struct BusyBodyScoped : NoAssign {
    tbb::enumerable_thread_specific<double> &locals;
    const int nThread;
    const int WorkRatiox100;
    int &unprotected_count;
    bool test_throw;

    BusyBodyScoped( int nThread_, int workRatiox100_, tbb::enumerable_thread_specific<double> &locals_, int &unprotected_count_, bool test_throw_) :
        locals(locals_),
        nThread(nThread_),
        WorkRatiox100(workRatiox100_),
        unprotected_count(unprotected_count_),
        test_throw(test_throw_) {
        sBarrier.initialize(nThread_);
    }

    void operator()(const int /* threadID */ ) const {
        int nIters = MAX_WORK/nThread;
        sBarrier.wait();
        tbb::tick_count t0 = tbb::tick_count::now();
        for(int j = 0; j < nIters; j++) {

            for(int i = 0; i < MAX_WORK * (100 - WorkRatiox100); i++) {
                locals.local() += 1.0;
            }
            {
                tbb::critical_section::scoped_lock my_lock(cs);
                for(int i = 0; i < MAX_WORK * WorkRatiox100; i++) {
                    locals.local() += 1.0;
                }
                unprotected_count++;
            }
        }
        locals.local() = (tbb::tick_count::now() - t0).seconds();
    }
};

void
RunOneCriticalSectionTest(int nThreads, int csWorkRatio, bool test_throw) {
    tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
    tbb::enumerable_thread_specific<double> test_locals;
    int myCount = 0;
    BusyBody myBody(nThreads, csWorkRatio, test_locals, myCount, test_throw);
    BusyBodyScoped myScopedBody(nThreads, csWorkRatio, test_locals, myCount, test_throw);
    init.initialize(nThreads);
    tbb::tick_count t0;
    {
        t0 = tbb::tick_count::now();
        myCount = 0;
        NativeParallelFor(nThreads, myBody);
        ASSERT(myCount == (MAX_WORK - (MAX_WORK % nThreads)), NULL);
        REMARK("%d threads, work ratio %d per cent, time %g", nThreads, csWorkRatio, (tbb::tick_count::now() - t0).seconds());
        if (nThreads > 1) {
            double etsSum = 0;
            double etsMax = 0;
            double etsMin = 0;
            double etsSigmaSq = 0;
            double etsSigma = 0;

            for(tbb::enumerable_thread_specific<double>::const_iterator ci = test_locals.begin(); ci != test_locals.end(); ci++) {
                etsSum += *ci;
                if(etsMax==0.0) {
                    etsMin = *ci;
                }
                else {
                    if(etsMin > *ci) etsMin = *ci;
                }
                if(etsMax < *ci) etsMax = *ci;
            }
            double etsAvg = etsSum / (double)nThreads;
            for(tbb::enumerable_thread_specific<double>::const_iterator ci = test_locals.begin(); ci != test_locals.end(); ci++) {
                etsSigma = etsAvg - *ci;
                etsSigmaSq += etsSigma * etsSigma;
            }
            // an attempt to gauge the "fairness" of the scheduling of the threads.  We figure
            // the standard deviation, and compare it with the maximum deviation from the
            // average time.  If the difference is 0 that means all threads finished in the same
            // amount of time.  If non-zero, the difference is divided by the time, and the
            // negative log is taken.  If > 2, then the difference is on the order of 0.01*t
            // where T is the average time.  We aritrarily define this as "fair."
            etsSigma = sqrt(etsSigmaSq/double(nThreads));
            etsMax -= etsAvg;  // max - a == delta1
            etsMin = etsAvg - etsMin;  // a - min == delta2
            if(etsMax < etsMin) etsMax = etsMin;
            etsMax -= etsSigma;
            // ASSERT(etsMax >= 0, NULL);  // shouldn't the maximum difference from the mean be > the stddev?
            etsMax = (etsMax > 0.0) ? etsMax : 0.0;  // possible rounding error
            double fairness = etsMax / etsAvg;
            if(fairness == 0.0) {
                fairness = 100.0;
            }
            else fairness = - log10(fairness);
            if(fairness > 2.0 ) {
                REMARK("  Fair (%g)\n", fairness);
            }
            else {
                REMARK("  Unfair (%g)\n", fairness);
            }
        }
        myCount = 0;
        NativeParallelFor(nThreads, myScopedBody);
        ASSERT(myCount == (MAX_WORK - (MAX_WORK % nThreads)), NULL);

    }

    init.terminate();
}

void
RunParallelTests() {
    for(int p = MinThread; p <= MaxThread; p++) {
        for(int cs_ratio = 1; cs_ratio < 95; cs_ratio *= 2) {
            RunOneCriticalSectionTest(p, cs_ratio, /*test_throw*/true);
        }
    }
}

int TestMain () {
    if(MinThread <= 0) MinThread = 1;

    if(MaxThread > 0) {
        RunParallelTests();
    }

    return Harness::Done;
}
