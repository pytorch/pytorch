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

#ifndef TBB_PREVIEW_AGGREGATOR
    #define TBB_PREVIEW_AGGREGATOR 1
#endif

#include "tbb/aggregator.h"
#include "harness.h"
#include <queue>

typedef std::priority_queue<int, std::vector<int>, std::less<int> > pq_t;

int N;
int* shared_data;

// Code for testing basic interface using function objects
class push_fnobj : NoAssign, Harness::NoAfterlife {
    pq_t& pq;
    int threadID;
public:
    push_fnobj(pq_t& pq_, int tid) : pq(pq_), threadID(tid) {}
    void operator()() const {
        AssertLive();
        pq.push(threadID);
    }
};

class pop_fnobj : NoAssign, Harness::NoAfterlife {
    pq_t& pq;
public:
    pop_fnobj(pq_t& pq_) : pq(pq_) {}
    void operator()() const {
        AssertLive();
        ASSERT(!pq.empty(), "queue should not be empty yet");
        int elem = pq.top();
        pq.pop();
        shared_data[elem]++;
    }
};

class BasicBody : NoAssign {
    pq_t& pq;
    tbb::aggregator& agg;
public:
    BasicBody(pq_t& pq_, tbb::aggregator& agg_) : pq(pq_), agg(agg_) {}
    void operator()(const int threadID) const {
        for (int i=0; i<N; ++i) agg.execute( push_fnobj(pq, threadID) );
        for (int i=0; i<N; ++i) agg.execute( pop_fnobj(pq) );
    }
};

void TestBasicInterface(int nThreads) {
    pq_t my_pq;
    tbb::aggregator agg;
    for (int i=0; i<MaxThread; ++i) shared_data[i] = 0;
    REMARK("Testing aggregator basic interface.\n");
    NativeParallelFor(nThreads, BasicBody(my_pq, agg));
    for (int i=0; i<nThreads; ++i)
        ASSERT(shared_data[i] == N, "wrong number of elements pushed");
    REMARK("Done testing aggregator basic interface.\n");
}
// End of code for testing basic interface using function objects


// Code for testing basic interface using lambda expressions
#if __TBB_CPP11_LAMBDAS_PRESENT
void TestBasicLambdaInterface(int nThreads) {
    pq_t my_pq;
    tbb::aggregator agg;
    for (int i=0; i<MaxThread; ++i) shared_data[i] = 0;
    REMARK("Testing aggregator basic lambda interface.\n");
    NativeParallelFor(nThreads, [&agg, &my_pq](const int threadID) {
        for (int i=0; i<N; ++i)
            agg.execute( [&, threadID]() { my_pq.push(threadID); } );
        for (int i=0; i<N; ++i) {
            agg.execute( [&]() {
                ASSERT(!my_pq.empty(), "queue should not be empty yet");
                int elem = my_pq.top();
                my_pq.pop();
                shared_data[elem]++;
            } );
        }
    } );
    for (int i=0; i<nThreads; ++i)
        ASSERT(shared_data[i] == N, "wrong number of elements pushed");
    REMARK("Done testing aggregator basic lambda interface.\n");
}
#endif /* __TBB_CPP11_LAMBDAS_PRESENT */
// End of code for testing basic interface using lambda expressions

// Code for testing expert interface
class op_data : public tbb::aggregator_operation, NoAssign {
public:
    const int tid;
    op_data(const int tid_=-1) : tbb::aggregator_operation(), tid(tid_) {}
};

class my_handler {
    pq_t *pq;
public:
    my_handler() {}
    my_handler(pq_t *pq_) : pq(pq_) {}
    void operator()(tbb::aggregator_operation* op_list) const {
        while (op_list) {
            op_data& request = static_cast<op_data&>(*op_list);
            op_list = op_list->next();
            request.start();
            if (request.tid >= 0) pq->push(request.tid);
            else {
                ASSERT(!pq->empty(), "queue should not be empty!");
                int elem = pq->top();
                pq->pop();
                shared_data[elem]++;
            }
            request.finish();
        }
    }
};

class ExpertBody : NoAssign {
    pq_t& pq;
    tbb::aggregator_ext<my_handler>& agg;
public:
    ExpertBody(pq_t& pq_, tbb::aggregator_ext<my_handler>& agg_) : pq(pq_), agg(agg_) {}
    void operator()(const int threadID) const {
        for (int i=0; i<N; ++i) {
            op_data to_push(threadID);
            agg.process( &to_push );
        }
        for (int i=0; i<N; ++i) {
            op_data to_pop;
            agg.process( &to_pop );
        }
    }
};

void TestExpertInterface(int nThreads) {
    pq_t my_pq;
    tbb::aggregator_ext<my_handler> agg((my_handler(&my_pq)));
    for (int i=0; i<MaxThread; ++i) shared_data[i] = 0;
    REMARK("Testing aggregator expert interface.\n");
    NativeParallelFor(nThreads, ExpertBody(my_pq, agg));
    for (int i=0; i<nThreads; ++i)
        ASSERT(shared_data[i] == N, "wrong number of elements pushed");
    REMARK("Done testing aggregator expert interface.\n");
}
// End of code for testing expert interface

int TestMain() {
    if (MinThread < 1)
        MinThread = 1;
    shared_data = new int[MaxThread];
    for (int p = MinThread; p <= MaxThread; ++p) {
        REMARK("Testing on %d threads.\n", p);
        N = 0;
        while (N <= 100) {
            REMARK("Testing with N=%d\n", N);
            TestBasicInterface(p);
#if __TBB_CPP11_LAMBDAS_PRESENT
            TestBasicLambdaInterface(p);
#endif
            TestExpertInterface(p);
            N = N ? N*10 : 1;
        }
    }
    return Harness::Done;
}
