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

#define HARNESS_DEFINE_PRIVATE_PUBLIC 1
#include "harness_inject_scheduler.h"
#define private public
#define protected public
#include "tbb/concurrent_queue.h"
#include "../tbb/concurrent_queue.cpp"
#undef protected
#undef private
#include "harness.h"

#if _MSC_VER==1500 && !__INTEL_COMPILER
    // VS2008/VC9 seems to have an issue; limits pull in math.h
    #pragma warning( push )
    #pragma warning( disable: 4985 )
#endif
#include <limits>
#if _MSC_VER==1500 && !__INTEL_COMPILER
    #pragma warning( pop )
#endif

template <typename Q>
class FloggerBody : NoAssign {
    Q& q;
    size_t elem_num;
public:
    FloggerBody(Q& q_, size_t elem_num_) : q(q_), elem_num(elem_num_) {}
    void operator()(const int threadID) const {
        typedef typename Q::value_type value_type;
        value_type elem = value_type(threadID);
        for (size_t i = 0; i < elem_num; ++i) {
            q.push(elem);
            (void) q.try_pop(elem);
        }
    }
};

template <typename Q>
void TestFloggerHelp(Q& q, size_t items_per_page) {
    size_t nq = q.my_rep->n_queue;
    size_t reserved_elem_num = nq * items_per_page - 1;
    size_t hack_val = std::numeric_limits<std::size_t>::max() & ~reserved_elem_num;
    q.my_rep->head_counter = hack_val;
    q.my_rep->tail_counter = hack_val;
    size_t k = q.my_rep->tail_counter & -(ptrdiff_t)nq;

    for (size_t i=0; i<nq; ++i) {
        q.my_rep->array[i].head_counter = k;
        q.my_rep->array[i].tail_counter = k;
    }
    NativeParallelFor(MaxThread, FloggerBody<Q>(q, reserved_elem_num + 20)); // to induce the overflow occurrence
    ASSERT(q.empty(), "FAILED flogger/empty test.");
    ASSERT(q.my_rep->head_counter < hack_val, "FAILED wraparound test.");
}

template <typename T>
void TestFlogger() {
    {
        tbb::concurrent_queue<T> q;
        REMARK("Wraparound on strict_ppl::concurrent_queue...");
        TestFloggerHelp(q, q.my_rep->items_per_page);
        REMARK(" works.\n");
    }
    {
        tbb::concurrent_bounded_queue<T> q;
        REMARK("Wraparound on tbb::concurrent_bounded_queue...");
        TestFloggerHelp(q, q.items_per_page);
        REMARK(" works.\n");
    }
}

void TestWraparound() {
    REMARK("Testing Wraparound...\n");
    TestFlogger<int>();
    TestFlogger<unsigned char>();
    REMARK("Done Testing Wraparound.\n");
}

int TestMain () {
    TestWraparound();
    return Harness::Done;
}
