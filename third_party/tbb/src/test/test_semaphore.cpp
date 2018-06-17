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

//
// Test for counting semaphore.
//
// set semaphore to N
// create N + M threads
// have each thread
//   A. P()
//   B. increment atomic count
//   C. spin for awhile checking the value of the count; make sure it doesn't exceed N
//   D. decrement atomic count
//   E. V()
//

#include "../tbb/semaphore.h"
#include "tbb/atomic.h"
#include "tbb/blocked_range.h"

#include <vector>
using std::vector;

#include "harness_assert.h"
#include "harness.h"

using tbb::internal::semaphore;

#include "harness_barrier.h"

tbb::atomic<int> pCount;

Harness::SpinBarrier sBarrier;

#include "tbb/tick_count.h"
// semaphore basic function:
//   set semaphore to initial value
//   see that semaphore only allows that number of threads to be active
class Body: NoAssign {
    const int nThreads;
    const int nIters;
    tbb::internal::semaphore &mySem;
    vector<int> &ourCounts;
    vector<double> &tottime;
    static const int tickCounts = 1;  // millisecond
    static const int innerWait = 5; // millisecond
public:
    Body(int nThread_, int nIter_, semaphore &mySem_,
            vector<int>& ourCounts_,
            vector<double>& tottime_
            ) : nThreads(nThread_), nIters(nIter_), mySem(mySem_), ourCounts(ourCounts_), tottime(tottime_) { sBarrier.initialize(nThread_); pCount = 0; }
void operator()(const int tid) const {
    sBarrier.wait();
    for(int i=0; i < nIters; ++i) {
        Harness::Sleep( tid * tickCounts );
        tbb::tick_count t0 = tbb::tick_count::now();
        mySem.P();
        tbb::tick_count t1 = tbb::tick_count::now();
        tottime[tid] += (t1-t0).seconds();
        int curval = ++pCount;
        if(curval > ourCounts[tid]) ourCounts[tid] = curval;
        Harness::Sleep( innerWait );
        --pCount;
        ASSERT((int)pCount >= 0, NULL);
        mySem.V();
    }
}
};


void testSemaphore( int semInitCnt, int extraThreads ) {
    semaphore my_sem(semInitCnt);
    // tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
    int nThreads = semInitCnt + extraThreads;
    vector<int> maxVals(nThreads);
    vector<double> totTimes(nThreads);
    int nIters = 10;
    Body myBody(nThreads, nIters, my_sem, maxVals, totTimes);

    REMARK( " sem(%d) with %d extra threads\n", semInitCnt, extraThreads);
    pCount = 0;
    NativeParallelFor(nThreads, myBody);
    if(extraThreads == 0) {
        double allPWaits = 0;
        for(vector<double>::const_iterator j = totTimes.begin(); j != totTimes.end(); ++j) {
            allPWaits += *j;
        }
        allPWaits /= static_cast<double>(nThreads * nIters);
        REMARK("Average wait for P() in uncontested case for nThreads = %d is %g\n", nThreads, allPWaits);
    }
    ASSERT(!pCount, "not all threads decremented pCount");
    int maxCount = -1;
    for(vector<int>::const_iterator i=maxVals.begin(); i!= maxVals.end();++i) {
        maxCount = max(maxCount,*i);
    }
    ASSERT(maxCount <= semInitCnt,"too many threads in semaphore-protected increment");
    if(maxCount < semInitCnt) {
        REMARK("Not enough threads in semaphore-protected region (%d < %d)\n", static_cast<int>(maxCount), semInitCnt);
    }
}

#include "../tbb/semaphore.cpp"
#if _WIN32||_WIN64
#include "../tbb/dynamic_link.cpp"

void testOSVersion() {
#if __TBB_USE_SRWLOCK
     BOOL bIsWindowsVistaOrLater;
#if  __TBB_WIN8UI_SUPPORT
     bIsWindowsVistaOrLater = true;
#else
     OSVERSIONINFO osvi;

     memset( (void*)&osvi, 0, sizeof(OSVERSIONINFO) );
     osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
     GetVersionEx(&osvi);
     bIsWindowsVistaOrLater = (osvi.dwMajorVersion >= 6 );
#endif

     if( bIsWindowsVistaOrLater ) {
        REMARK("Checking SRWLock is loaded\n");
        tbb::internal::binary_semaphore s;
        ASSERT( (uintptr_t)tbb::internal::__TBB_init_binsem!=(uintptr_t)&tbb::internal::init_binsem_using_event, NULL );
        ASSERT( (uintptr_t)tbb::internal::__TBB_acquire_binsem!=(uintptr_t)&tbb::internal::acquire_binsem_using_event, NULL );
        ASSERT( (uintptr_t)tbb::internal::__TBB_release_binsem!=(uintptr_t)&tbb::internal::release_binsem_using_event, NULL );
     }
#endif /* __TBB_USE_SRWLOCK */
}
#endif /* _WIN32||_WIN64 */

#define N_TIMES 1000

template<typename S>
struct Counter {
    volatile long value;
    S my_sem;
    Counter() : value(0) {}
};

//! Function object for use with parallel_for.h.
template<typename C>
struct AddOne: NoAssign {
    C& my_counter;
    /** Increments counter once for each iteration in the iteration space. */
    void operator()( int /*tid*/ ) const {
        for( size_t i=0; i<N_TIMES; ++i ) {
            my_counter.my_sem.P();
            my_counter.value = my_counter.value + 1;
            my_counter.my_sem.V();
        }
    }
    AddOne( C& c_ ) : my_counter(c_) { my_counter.my_sem.V(); }
};

void testBinarySemaphore( int nThreads ) {
    REMARK("Testing binary semaphore\n");
    Counter<tbb::internal::binary_semaphore> counter;
    AddOne<Counter<tbb::internal::binary_semaphore> > myAddOne(counter);
    NativeParallelFor( nThreads, myAddOne );
    ASSERT( nThreads*N_TIMES==counter.value, "Binary semaphore operations P()/V() have a race");
}

// Power of 2, the most tokens that can be in flight.
#define MAX_TOKENS 32
enum FilterType { imaProducer, imaConsumer };
class FilterBase : NoAssign {
protected:
    FilterType ima;
    unsigned totTokens;  // total number of tokens to be emitted, only used by producer
    tbb::atomic<unsigned>& myTokens;
    tbb::atomic<unsigned>& otherTokens;
    unsigned myWait;
    semaphore &mySem;
    semaphore &nextSem;
    unsigned* myBuffer;
    unsigned* nextBuffer;
    unsigned curToken;
public:
    FilterBase( FilterType ima_
            ,unsigned totTokens_
            ,tbb::atomic<unsigned>& myTokens_
            ,tbb::atomic<unsigned>& otherTokens_
            ,unsigned myWait_
            ,semaphore &mySem_
            ,semaphore &nextSem_
            ,unsigned* myBuffer_
            ,unsigned* nextBuffer_
            )
        : ima(ima_),totTokens(totTokens_),myTokens(myTokens_),otherTokens(otherTokens_),myWait(myWait_),mySem(mySem_),
          nextSem(nextSem_),myBuffer(myBuffer_),nextBuffer(nextBuffer_)
    {
        curToken = 0;
    }
    void Produce(const int tid);
    void Consume(const int tid);
    void operator()(const int tid) { if(ima == imaConsumer) Consume(tid); else Produce(tid); }
};

class ProduceConsumeBody {
    FilterBase** myFilters;
    public:
    ProduceConsumeBody(FilterBase** myFilters_) : myFilters(myFilters_) {}
    void operator()(const int tid) const {
        myFilters[tid]->operator()(tid);
    }
};

// send a bunch of non-Null "tokens" to consumer, then a NULL.
void FilterBase::Produce(const int /*tid*/) {
    nextBuffer[0] = 0;  // just in case we provide no tokens
    sBarrier.wait();
    while(totTokens) {
        while(!myTokens)
            mySem.P();
        // we have a slot available.
        --myTokens;  // moving this down reduces spurious wakeups
        --totTokens;
        if(totTokens)
            nextBuffer[curToken&(MAX_TOKENS-1)] = curToken*3+1;
        else
            nextBuffer[curToken&(MAX_TOKENS-1)] = 0;
        ++curToken;
        Harness::Sleep(myWait);
        unsigned temp = ++otherTokens;
        if(temp == 1)
            nextSem.V();
    }
    nextSem.V();  // final wakeup
}

void FilterBase::Consume(const int /*tid*/) {
    unsigned myToken;
    sBarrier.wait();
    do {
        while(!myTokens)
            mySem.P();
        // we have a slot available.
        --myTokens;  // moving this down reduces spurious wakeups
        myToken = myBuffer[curToken&(MAX_TOKENS-1)];
        if(myToken) {
            ASSERT(myToken == curToken*3+1, "Error in received token");
            ++curToken;
            Harness::Sleep(myWait);
            unsigned temp = ++otherTokens;
            if(temp == 1)
                nextSem.V();
        }
    } while(myToken);
    // end of processing
    ASSERT(curToken + 1 == totTokens, "Didn't receive enough tokens");
}

// -- test of producer/consumer with atomic buffer cnt and semaphore
// nTokens are total number of tokens through the pipe
// pWait is the wait time for the producer
// cWait is the wait time for the consumer
void testProducerConsumer( unsigned totTokens, unsigned nTokens, unsigned pWait, unsigned cWait) {
    semaphore pSem;
    semaphore cSem;
    tbb::atomic<unsigned> pTokens;
    tbb::atomic<unsigned> cTokens;
    cTokens = 0;
    unsigned cBuffer[MAX_TOKENS];
    FilterBase* myFilters[2];  // one producer, one consumer
    REMARK("Testing producer/consumer with %lu total tokens, %lu tokens at a time, producer wait(%lu), consumer wait (%lu)\n", totTokens, nTokens, pWait, cWait);
    ASSERT(nTokens <= MAX_TOKENS, "Not enough slots for tokens");
    myFilters[0] = new FilterBase(imaProducer, totTokens, pTokens, cTokens, pWait, cSem, pSem, (unsigned *)NULL, &(cBuffer[0]));
    myFilters[1] = new FilterBase(imaConsumer, totTokens, cTokens, pTokens, cWait, pSem, cSem, cBuffer, (unsigned *)NULL);
    pTokens = nTokens;
    ProduceConsumeBody myBody(myFilters);
    sBarrier.initialize(2);
    NativeParallelFor(2, myBody);
    delete myFilters[0];
    delete myFilters[1];
}

int TestMain() {
    REMARK("Started\n");
#if _WIN32||_WIN64
    testOSVersion();
#endif
    if(MaxThread > 0) {
        testBinarySemaphore( MaxThread );
        for(int semSize = 1; semSize <= MaxThread; ++semSize) {
            for(int exThreads = 0; exThreads <= MaxThread - semSize; ++exThreads) {
                testSemaphore( semSize, exThreads );
            }
        }
    }
    // Test producer/consumer with varying execution times and buffer sizes
    // ( total tokens, tokens in buffer, sleep for producer, sleep for consumer )
    testProducerConsumer( 10, 2, 5, 5 );
    testProducerConsumer( 10, 2, 20, 5 );
    testProducerConsumer( 10, 2, 5, 20 );
    testProducerConsumer( 10, 1, 5, 5 );
    testProducerConsumer( 20, 10, 5, 20 );
    testProducerConsumer( 64, 32, 1, 20 );
    return Harness::Done;
}
