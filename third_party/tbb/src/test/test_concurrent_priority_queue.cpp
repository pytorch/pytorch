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

#include "harness_defs.h"
#include "tbb/concurrent_priority_queue.h"
#include "tbb/atomic.h"
#include "tbb/blocked_range.h"
#include "harness.h"
#include <functional>
#include <algorithm>
#include "harness_allocator.h"
#include <vector>
#include "test_container_move_support.h"

#if _MSC_VER==1500 && !__INTEL_COMPILER
    // VS2008/VC9 seems to have an issue; limits pull in math.h
    #pragma warning( push )
    #pragma warning( disable: 4985 )
#endif
#include <climits>
#if _MSC_VER==1500 && !__INTEL_COMPILER
    #pragma warning( pop )
#endif

#if __INTEL_COMPILER && (_WIN32 || _WIN64) && TBB_USE_DEBUG && _CPPLIB_VER<520
// The Intel Compiler has an issue that causes the Microsoft Iterator
// Debugging code to crash in vector::pop_back when it is called after a
// vector::push_back throws an exception.
// #define _HAS_ITERATOR_DEBUGGING 0 // Setting this to 0 doesn't solve the problem
                                     // and also provokes a redefinition warning
#define __TBB_ITERATOR_DEBUGGING_EXCEPTIONS_BROKEN
#endif

using namespace tbb;

const size_t MAX_ITER = 10000;

tbb::atomic<unsigned int> counter;

class my_data_type {
public:
    int priority;
    char padding[tbb::internal::NFS_MaxLineSize - sizeof(int) % tbb::internal::NFS_MaxLineSize];
    my_data_type() {}
    my_data_type(int init_val) : priority(init_val) {}
    const my_data_type operator+(const my_data_type& other) const {
        return my_data_type(priority+other.priority);
    }
    bool operator==(const my_data_type& other) const {
        return this->priority == other.priority;
    }
};

const my_data_type DATA_MIN(INT_MIN);
const my_data_type DATA_MAX(INT_MAX);

class my_less {
public:
    bool operator()(const my_data_type d1, const my_data_type d2) const {
        return d1.priority<d2.priority;
    }
};

#if TBB_USE_EXCEPTIONS
class my_throwing_type : public my_data_type {
public:
    static int throw_flag;
    my_throwing_type() : my_data_type() {}
    my_throwing_type(const my_throwing_type& src) : my_data_type(src) {
        if (my_throwing_type::throw_flag) throw 42;
        priority = src.priority;
    }
};
int my_throwing_type::throw_flag = 0;

typedef concurrent_priority_queue<my_throwing_type, my_less > cpq_ex_test_type;
#endif

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT
const size_t push_selector_variants = 3;
#elif __TBB_CPP11_RVALUE_REF_PRESENT
const size_t push_selector_variants = 2;
#else
const size_t push_selector_variants = 1;
#endif

template <typename Q, typename E>
void push_selector(Q& q, E e, size_t i) {
    switch (i%push_selector_variants) {
    case 0: q->push(e); break;
#if __TBB_CPP11_RVALUE_REF_PRESENT
    case 1: q->push(tbb::internal::move(e)); break;
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    case 2: q->emplace(e); break;
#endif
#endif
    }
}

template<typename T, typename C>
class FillBody : NoAssign {
    int nThread;
    T my_max, my_min;
    concurrent_priority_queue<T, C> *q;
    C less_than;
public:
    FillBody(int nThread_, T max_, T min_, concurrent_priority_queue<T, C> *q_) : nThread(nThread_), my_max(max_), my_min(min_), q(q_) {}
    void operator()(const int threadID) const {
        T elem = my_min + T(threadID);
        for (size_t i=0; i<MAX_ITER; ++i) {
            // do some pushes
            push_selector(q, elem, i);
            if (elem == my_max) elem = my_min;
            elem = elem + T(nThread);
        }
    }
};

template<typename T, typename C>
struct EmptyBody : NoAssign {
    int nThread;
    T my_max;
    concurrent_priority_queue<T, C> *q;
    C less_than;
public:
    EmptyBody(int nThread_, T max_, concurrent_priority_queue<T, C> *q_) : nThread(nThread_), my_max(max_), q(q_) {}
    void operator()(const int /*threadID*/) const {
        T elem(my_max), last;
        if (q->try_pop(last)) {
            ++counter;
            while(q->try_pop(elem)) {
                ASSERT(!less_than(last, elem), "FAILED pop/priority test in EmptyBody.");
                last = elem;
                elem = my_max;
                ++counter;
            }
        }
    }
};

template <typename T, typename C>
class FloggerBody : NoAssign {
    int nThread;
    concurrent_priority_queue<T, C> *q;
public:
    FloggerBody(int nThread_, concurrent_priority_queue<T, C> *q_) :
        nThread(nThread_), q(q_) {}
    void operator()(const int threadID) const {
        T elem = T(threadID+1);
        for (size_t i=0; i<MAX_ITER; ++i) {
            push_selector(q, elem, i);
            (void) q->try_pop(elem);
        }
    }
};

namespace equality_comparison_helpers {
    struct to_vector{
        template <typename element_type, typename compare_t, typename allocator_t>
        std::vector<element_type> operator()(tbb::concurrent_priority_queue<element_type, compare_t, allocator_t> const& source) const{
            tbb::concurrent_priority_queue<element_type, compare_t, allocator_t>  cpq((source));
            std::vector<element_type> v; v.reserve(cpq.size());
            element_type element;
            while (cpq.try_pop(element)){ v.push_back(element);}
            std::reverse(v.begin(),v.end());
            return v;
        }
    };
}
//TODO: make CPQ more testable instead of hacking ad-hoc operator ==
//operator == is required for __TBB_TEST_INIT_LIST_SUITE
template <typename element_type, typename compare_t, typename allocator_t>
bool operator==(tbb::concurrent_priority_queue<element_type, compare_t, allocator_t> const& lhs, tbb::concurrent_priority_queue<element_type, compare_t, allocator_t> const& rhs){
    using equality_comparison_helpers::to_vector;
    return to_vector()(lhs) == to_vector()(rhs);
}

template <typename range, typename element_type, typename compare_t, typename allocator_t>
bool operator==(tbb::concurrent_priority_queue<element_type, compare_t, allocator_t> const& lhs, range const & rhs ){
    using equality_comparison_helpers::to_vector;
    return to_vector()(lhs) == std::vector<element_type>(rhs.begin(),rhs.end());
}

void TestToVector(){
    using equality_comparison_helpers::to_vector;
    int array[] = {1,5,6,8,4,7};
    tbb::blocked_range<int *> range =  Harness::make_blocked_range(array);
    std::vector<int> source(range.begin(),range.end());
    tbb::concurrent_priority_queue<int> q(source.begin(),source.end());
    std::vector<int> from_cpq = to_vector()(q);
    std::sort(source.begin(),source.end());
    ASSERT(source == from_cpq,"equality_comparison_helpers::to_vector incorrectly copied items from CPQ?");
}

void TestHelpers(){
    TestToVector();
}

void TestConstructorsDestructorsAccessors() {
    std::vector<int> v;
    std::allocator<int> a;
    concurrent_priority_queue<int, std::less<int> > *q, *qo;
    concurrent_priority_queue<int, std::less<int>, std::allocator<int>  > *qi;

    // Test constructors/destructors
    REMARK("Testing default constructor.\n");
    q = new concurrent_priority_queue<int, std::less<int> >();
    REMARK("Default constructor complete.\n");
    ASSERT(q->size()==0, "FAILED size test.");
    ASSERT(q->empty(), "FAILED empty test.");
    REMARK("Testing destructor.\n");
    delete q;
    REMARK("Destruction complete.\n");

    REMARK("Testing capacity constructor.\n");
    q = new concurrent_priority_queue<int, std::less<int> >(42);
    REMARK("Capacity constructor complete.\n");
    ASSERT(q->size()==0, "FAILED size test.");
    ASSERT(q->empty(), "FAILED empty test.");
    REMARK("Testing destructor.\n");
    delete q;
    REMARK("Destruction complete.\n");

    REMARK("Testing allocator constructor.\n");
    qi = new concurrent_priority_queue<int, std::less<int>, std::allocator<int> >(a);
    REMARK("Allocator constructor complete.\n");
    ASSERT(qi->size()==0, "FAILED size test.");
    ASSERT(qi->empty(), "FAILED empty test.");
    REMARK("Testing destructor.\n");
    delete qi;
    REMARK("Destruction complete.\n");

    REMARK("Testing capacity+allocator constructor.\n");
    qi = new concurrent_priority_queue<int, std::less<int>, std::allocator<int> >(42, a);
    REMARK("Capacity+allocator constructor complete.\n");
    ASSERT(qi->size()==0, "FAILED size test.");
    ASSERT(qi->empty(), "FAILED empty test.");
    REMARK("Testing destructor.\n");
    delete qi;
    REMARK("Destruction complete.\n");

    REMARK("Testing iterator filler constructor.\n");
    for (int i=0; i<42; ++i)
        v.push_back(i);
    q = new concurrent_priority_queue<int, std::less<int> >(v.begin(), v.end());
    REMARK("Iterator filler constructor complete.\n");
    ASSERT(q->size()==42, "FAILED vector/size test.");
    ASSERT(!q->empty(), "FAILED vector/empty test.");
    ASSERT(*q == v, "FAILED vector/equality test.");

    REMARK("Testing copy constructor.\n");
    qo = new concurrent_priority_queue<int, std::less<int> >(*q);
    REMARK("Copy constructor complete.\n");
    ASSERT(qo->size()==42, "FAILED cpq/size test.");
    ASSERT(!qo->empty(), "FAILED cpq/empty test.");
    ASSERT(*q == *qo, "FAILED cpq/equality test.");

    REMARK("Testing destructor.\n");
    delete q;
    delete qo;
    REMARK("Destruction complete.\n");
}

void TestAssignmentClearSwap() {
    typedef concurrent_priority_queue<int, std::less<int> > cpq_type;
    std::vector<int> v;
    cpq_type *q, *qo;
    int e;

    for (int i=0; i<42; ++i)
        v.push_back(i);
    q = new cpq_type(v.begin(), v.end());
    qo = new cpq_type();

    REMARK("Testing assignment (1).\n");
    *qo = *q;
    REMARK("Assignment complete.\n");
    ASSERT(qo->size()==42, "FAILED assignment/size test.");
    ASSERT(!qo->empty(), "FAILED assignment/empty test.");
    ASSERT(*qo == v,"FAILED assignment/equality test");

    cpq_type assigned_q;
    REMARK("Testing assign(begin,end) (2).\n");
    assigned_q.assign(v.begin(), v.end());
    REMARK("Assignment complete.\n");
    ASSERT(assigned_q.size()==42, "FAILED assignment/size test.");
    ASSERT(!assigned_q.empty(), "FAILED assignment/empty test.");
    ASSERT(assigned_q == v,"FAILED assignment/equality test");

    REMARK("Testing clear.\n");
    q->clear();
    REMARK("Clear complete.\n");
    ASSERT(q->size()==0, "FAILED clear/size test.");
    ASSERT(q->empty(), "FAILED clear/empty test.");

    for (size_t i=0; i<5; ++i)
        (void) qo->try_pop(e);

    REMARK("Testing assignment (3).\n");
    *q = *qo;
    REMARK("Assignment complete.\n");
    ASSERT(q->size()==37, "FAILED assignment/size test.");
    ASSERT(!q->empty(), "FAILED assignment/empty test.");

    for (size_t i=0; i<5; ++i)
        (void) qo->try_pop(e);

    REMARK("Testing swap.\n");
    q->swap(*qo);
    REMARK("Swap complete.\n");
    ASSERT(q->size()==32, "FAILED swap/size test.");
    ASSERT(!q->empty(), "FAILED swap/empty test.");
    ASSERT(qo->size()==37, "FAILED swap_operand/size test.");
    ASSERT(!qo->empty(), "FAILED swap_operand/empty test.");
    delete q;
    delete qo;
}

void TestSerialPushPop() {
    concurrent_priority_queue<int, std::less<int> > *q;
    int e=42, prev=INT_MAX;
    size_t count=0;

    q = new concurrent_priority_queue<int, std::less<int> >(MAX_ITER);
    REMARK("Testing serial push.\n");
    for (size_t i=0; i<MAX_ITER; ++i) {
        push_selector(q, e, i);
        e = e*-1 + int(i);
    }
    REMARK("Pushing complete.\n");
    ASSERT(q->size()==MAX_ITER, "FAILED push/size test.");
    ASSERT(!q->empty(), "FAILED push/empty test.");

    REMARK("Testing serial pop.\n");
    while (!q->empty()) {
        ASSERT(q->try_pop(e), "FAILED pop test.");
        ASSERT(prev>=e, "FAILED pop/priority test.");
        prev = e;
        ++count;
        ASSERT(q->size()==MAX_ITER-count, "FAILED swap/size test.");
        ASSERT(!q->empty() || count==MAX_ITER, "FAILED swap/empty test.");
    }
    ASSERT(!q->try_pop(e), "FAILED: successful pop from the empty queue.");
    REMARK("Popping complete.\n");
    delete q;
}

template <typename T, typename C>
void TestParallelPushPop(int nThreads, T t_max, T t_min, C /*compare*/) {
    size_t qsize;

    concurrent_priority_queue<T, C> *q = new concurrent_priority_queue<T, C>(0);
    FillBody<T, C> filler(nThreads, t_max, t_min, q);
    EmptyBody<T, C> emptier(nThreads, t_max, q);
    counter = 0;
    REMARK("Testing parallel push.\n");
    NativeParallelFor(nThreads, filler);
    REMARK("Pushing complete.\n");
    qsize = q->size();
    ASSERT(q->size()==nThreads*MAX_ITER, "FAILED push/size test.");
    ASSERT(!q->empty(), "FAILED push/empty test.");

    REMARK("Testing parallel pop.\n");
    NativeParallelFor(nThreads, emptier);
    REMARK("Popping complete.\n");
    ASSERT(counter==qsize, "FAILED pop/size test.");
    ASSERT(q->size()==0, "FAILED pop/empty test.");

    q->clear();
    delete(q);
}

void TestExceptions() {
#if TBB_USE_EXCEPTIONS
    const size_t TOO_LARGE_SZ = 1000000000;
    my_throwing_type elem;

    REMARK("Testing basic constructor exceptions.\n");
    // Allocate empty queue should not throw no matter the type
    try {
        my_throwing_type::throw_flag = 1;
        cpq_ex_test_type q;
    } catch(...) {
#if !(_MSC_VER==1900)
        ASSERT(false, "FAILED: allocating empty queue should not throw exception.\n");
        // VS2015 warns about the code in this catch block being unreachable
#endif
    }
    // Allocate small queue should not throw for reasonably sized type
    try {
        my_throwing_type::throw_flag = 1;
        cpq_ex_test_type q(42);
    } catch(...) {
        ASSERT(false, "FAILED: allocating small queue should not throw exception.\n");
    }
    // Allocate a queue with too large initial size
    try {
        my_throwing_type::throw_flag = 0;
        cpq_ex_test_type q(TOO_LARGE_SZ);
        REMARK("FAILED: Huge queue did not throw exception.\n");
    } catch(...) {}

    cpq_ex_test_type *pq;
    try {
        my_throwing_type::throw_flag = 0;
        pq = NULL;
        pq = new cpq_ex_test_type(TOO_LARGE_SZ);
        REMARK("FAILED: Huge queue did not throw exception.\n");
        delete pq;
    } catch(...) {
        ASSERT(!pq, "FAILED: pq should not be touched when constructor throws.\n");
    }
    REMARK("Basic constructor exceptions testing complete.\n");
    REMARK("Testing copy constructor exceptions.\n");
    my_throwing_type::throw_flag = 0;
    cpq_ex_test_type src_q(42);
    elem.priority = 42;
    for (size_t i=0; i<42; ++i) src_q.push(elem);
    try {
        my_throwing_type::throw_flag = 1;
        cpq_ex_test_type q(src_q);
        REMARK("FAILED: Copy construct did not throw exception.\n");
    } catch(...) {}
    try {
        my_throwing_type::throw_flag = 1;
        pq = NULL;
        pq = new concurrent_priority_queue<my_throwing_type, my_less >(src_q);
        REMARK("FAILED: Copy construct did not throw exception.\n");
        delete pq;
    } catch(...) {
        ASSERT(!pq, "FAILED: pq should not be touched when constructor throws.\n");
    }
    REMARK("Copy constructor exceptions testing complete.\n");
    REMARK("Testing assignment exceptions.\n");
    // Assignment is copy-swap, so it should be exception safe
    my_throwing_type::throw_flag = 0;
    cpq_ex_test_type assign_q(24);
    try {
        my_throwing_type::throw_flag = 1;
        assign_q = src_q;
        REMARK("FAILED: Assign did not throw exception.\n");
    } catch(...) {
        ASSERT(assign_q.empty(), "FAILED: assign_q should be empty.\n");
    }
    REMARK("Assignment exceptions testing complete.\n");
#ifndef __TBB_ITERATOR_DEBUGGING_EXCEPTIONS_BROKEN
    REMARK("Testing push exceptions.\n");
    for (size_t i=0; i<push_selector_variants; ++i) {
        my_throwing_type::throw_flag = 0;
        pq = new cpq_ex_test_type(3);
        try {
            push_selector(pq, elem, i);
            push_selector(pq, elem, i);
            push_selector(pq, elem, i);
        } catch(...) {
            ASSERT(false, "FAILED: Push should not throw exception... yet.\n");
        }
        try { // should crash on copy during expansion of vector
            my_throwing_type::throw_flag = 1;
            push_selector(pq, elem, i);
            REMARK("FAILED: Push did not throw exception.\n");
        } catch(...) {
            ASSERT(!pq->empty(), "FAILED: pq should not be empty.\n");
            ASSERT(pq->size()==3, "FAILED: pq should be only three elements.\n");
            ASSERT(pq->try_pop(elem), "FAILED: pq is not functional.\n");
        }
        delete pq;

        my_throwing_type::throw_flag = 0;
        pq = new cpq_ex_test_type(3);
        try {
            push_selector(pq, elem, i);
            push_selector(pq, elem, i);
        } catch(...) {
            ASSERT(false, "FAILED: Push should not throw exception... yet.\n");
        }
        try { // should crash on push copy of element
            my_throwing_type::throw_flag = 1;
            push_selector(pq, elem, i);
            REMARK("FAILED: Push did not throw exception.\n");
        } catch(...) {
            ASSERT(!pq->empty(), "FAILED: pq should not be empty.\n");
            ASSERT(pq->size()==2, "FAILED: pq should be only two elements.\n");
            ASSERT(pq->try_pop(elem), "FAILED: pq is not functional.\n");
        }
        delete pq;
    }
    REMARK("Push exceptions testing complete.\n");
#endif
#endif // TBB_USE_EXCEPTIONS
}

template <typename T, typename C>
void TestFlogger(int nThreads, T /*max*/, C /*compare*/) {
    REMARK("Testing queue flogger.\n");
    concurrent_priority_queue<T, C> *q = new concurrent_priority_queue<T, C> (0);
    NativeParallelFor(nThreads, FloggerBody<T, C >(nThreads, q));
    ASSERT(q->empty(), "FAILED flogger/empty test.");
    ASSERT(!q->size(), "FAILED flogger/size test.");
    REMARK("Flogging complete.\n");
    delete q;
}

#if __TBB_INITIALIZER_LISTS_PRESENT
#include "test_initializer_list.h"

void TestInitList(){
    REMARK("testing initializer_list methods \n");
    using namespace initializer_list_support_tests;
    TestInitListSupport<tbb::concurrent_priority_queue<char> >({1,2,3,4,5});
    TestInitListSupport<tbb::concurrent_priority_queue<int> >({});
}
#endif //if __TBB_INITIALIZER_LISTS_PRESENT

struct special_member_calls_t {
    size_t copy_constructor_called_times;
    size_t move_constructor_called_times;
    size_t copy_assignment_called_times;
    size_t move_assignment_called_times;

    bool friend operator==(special_member_calls_t const& lhs, special_member_calls_t const& rhs){
        return
                lhs.copy_constructor_called_times == rhs.copy_constructor_called_times
             && lhs.move_constructor_called_times == rhs.move_constructor_called_times
             && lhs.copy_assignment_called_times == rhs.copy_assignment_called_times
             && lhs.move_assignment_called_times == rhs.move_assignment_called_times;
    }

};
#if __TBB_CPP11_RVALUE_REF_PRESENT
struct MoveOperationTracker {
    static size_t copy_constructor_called_times;
    static size_t move_constructor_called_times;
    static size_t copy_assignment_called_times;
    static size_t move_assignment_called_times;

    static special_member_calls_t special_member_calls(){
        special_member_calls_t calls = {copy_constructor_called_times, move_constructor_called_times, copy_assignment_called_times, move_assignment_called_times};
        return calls;
    }
    static size_t value_counter;

    size_t value;

    MoveOperationTracker() : value(++value_counter) {}
    MoveOperationTracker( const size_t value_ ) : value( value_ ) {}
    ~MoveOperationTracker() __TBB_NOEXCEPT( true ) {
        value = 0;
    }
    MoveOperationTracker(const MoveOperationTracker& m) : value(m.value) {
        ASSERT(m.value, "The object has been moved or destroyed");
        ++copy_constructor_called_times;
    }
    MoveOperationTracker(MoveOperationTracker&& m) __TBB_NOEXCEPT(true) : value(m.value) {
        ASSERT(m.value, "The object has been moved or destroyed");
        m.value = 0;
        ++move_constructor_called_times;
    }
    MoveOperationTracker& operator=(MoveOperationTracker const& m) {
        ASSERT(m.value, "The object has been moved or destroyed");
        value = m.value;
        ++copy_assignment_called_times;
        return *this;
    }
    MoveOperationTracker& operator=(MoveOperationTracker&& m) __TBB_NOEXCEPT(true) {
        ASSERT(m.value, "The object has been moved or destroyed");
        value = m.value;
        m.value = 0;
        ++move_assignment_called_times;
        return *this;
    }

    bool operator<(MoveOperationTracker const &m) const {
        ASSERT(value, "The object has been moved or destroyed");
        ASSERT(m.value, "The object has been moved or destroyed");
        return value < m.value;
    }

    friend bool operator==(MoveOperationTracker const &lhs, MoveOperationTracker const &rhs){
        return !(lhs < rhs) && !(rhs <lhs);
    }
};
size_t MoveOperationTracker::copy_constructor_called_times = 0;
size_t MoveOperationTracker::move_constructor_called_times = 0;
size_t MoveOperationTracker::copy_assignment_called_times = 0;
size_t MoveOperationTracker::move_assignment_called_times = 0;
size_t MoveOperationTracker::value_counter = 0;

template<typename allocator = tbb::cache_aligned_allocator<MoveOperationTracker> >
struct cpq_src_fixture : NoAssign {
    enum {default_container_size = 100};
    typedef concurrent_priority_queue<MoveOperationTracker, std::less<MoveOperationTracker>, typename allocator:: template rebind<MoveOperationTracker>::other > cpq_t;

    cpq_t cpq_src;
    const size_t  container_size;

    void init(){
        size_t &mcct = MoveOperationTracker::move_constructor_called_times;
        size_t &ccct = MoveOperationTracker::copy_constructor_called_times;
        size_t &cact = MoveOperationTracker::copy_assignment_called_times;
        size_t &mact = MoveOperationTracker::move_assignment_called_times;
        mcct = ccct = cact = mact = 0;

        for (size_t i=1; i <= container_size; ++i){
            cpq_src.push(MoveOperationTracker(i));
        }
        ASSERT(cpq_src.size() == container_size, "error in test setup ?" );
    }

    cpq_src_fixture(size_t size = default_container_size) : container_size(size){
        init();
    }

    cpq_src_fixture(typename cpq_t::allocator_type const& a, size_t size = default_container_size) : cpq_src(a), container_size(size){
        init();
    }

};


void TestStealingMoveConstructor(){
    typedef cpq_src_fixture<> fixture_t;
    fixture_t fixture;
    fixture_t::cpq_t src_copy(fixture.cpq_src);

    special_member_calls_t previous = MoveOperationTracker::special_member_calls();
    fixture_t::cpq_t dst(std::move(fixture.cpq_src));
    ASSERT(previous == MoveOperationTracker::special_member_calls(), "stealing move constructor should not create any new elements");

    ASSERT(dst == src_copy, "cpq content changed during stealing move ?");
}

void TestStealingMoveConstructorOtherAllocatorInstance(){
    typedef two_memory_arenas_fixture<MoveOperationTracker> arena_fixture_t;
    typedef cpq_src_fixture<arena_fixture_t::allocator_t > fixture_t;

    arena_fixture_t arena_fixture(8 * fixture_t::default_container_size, "TestStealingMoveConstructorOtherAllocatorInstance");
    fixture_t fixture(arena_fixture.source_allocator);
    fixture_t::cpq_t src_copy(fixture.cpq_src);

    special_member_calls_t previous = MoveOperationTracker::special_member_calls();
    fixture_t::cpq_t dst(std::move(fixture.cpq_src), arena_fixture.source_allocator);
    ASSERT(previous == MoveOperationTracker::special_member_calls(), "stealing move constructor should not create any new elements");

    ASSERT(dst == src_copy, "cpq content changed during stealing move ?");
}

void TestPerElementMoveConstructorOtherAllocatorInstance(){
    typedef two_memory_arenas_fixture<MoveOperationTracker> arena_fixture_t;
    typedef cpq_src_fixture<arena_fixture_t::allocator_t > fixture_t;

    arena_fixture_t arena_fixture(8 * fixture_t::default_container_size, "TestPerElementMoveConstructorOtherAllocatorInstance");
    fixture_t fixture(arena_fixture.source_allocator);
    fixture_t::cpq_t src_copy(fixture.cpq_src);

    special_member_calls_t move_ctor_called_cpq_size_times = MoveOperationTracker::special_member_calls();
    move_ctor_called_cpq_size_times.move_constructor_called_times += fixture.container_size;

    fixture_t::cpq_t dst(std::move(fixture.cpq_src), arena_fixture.dst_allocator);
    ASSERT(move_ctor_called_cpq_size_times == MoveOperationTracker::special_member_calls(), "Per element move constructor should move initialize all new elements");
    ASSERT(dst == src_copy, "cpq content changed during move ?");
}

void TestgMoveConstructor(){
    TestStealingMoveConstructor();
    TestStealingMoveConstructorOtherAllocatorInstance();
    TestPerElementMoveConstructorOtherAllocatorInstance();
}

void TestStealingMoveAssignOperator(){
    typedef cpq_src_fixture<> fixture_t;
    fixture_t fixture;
    fixture_t::cpq_t src_copy(fixture.cpq_src);

    fixture_t::cpq_t dst;
    special_member_calls_t previous = MoveOperationTracker::special_member_calls();
    dst = std::move(fixture.cpq_src);
    ASSERT(previous == MoveOperationTracker::special_member_calls(), "stealing move assign operator should not create any new elements");

    ASSERT(dst == src_copy, "cpq content changed during stealing move ?");
}

void TestStealingMoveAssignOperatorWithStatefulAllocator(){
    //Use stateful allocator which is propagated on assignment , i.e. POCMA = true
    typedef two_memory_arenas_fixture<MoveOperationTracker, /*pocma =*/Harness::true_type> arena_fixture_t;
    typedef cpq_src_fixture<arena_fixture_t::allocator_t > fixture_t;

    arena_fixture_t arena_fixture(8 * fixture_t::default_container_size, "TestStealingMoveAssignOperatorWithStatefullAllocator");
    fixture_t fixture(arena_fixture.source_allocator);
    fixture_t::cpq_t src_copy(fixture.cpq_src);
    fixture_t::cpq_t dst(arena_fixture.dst_allocator);

    special_member_calls_t previous = MoveOperationTracker::special_member_calls();
    dst = std::move(fixture.cpq_src);
    ASSERT(previous == MoveOperationTracker::special_member_calls(), "stealing move assignment operator should not create any new elements");

    ASSERT(dst == src_copy, "cpq content changed during stealing move ?");
}

void TestPerElementMoveAssignOperator(){
    //use stateful allocator which is not propagate on assignment , i.e. POCMA = false
    typedef two_memory_arenas_fixture<MoveOperationTracker, /*pocma =*/Harness::false_type> arena_fixture_t;
    typedef cpq_src_fixture<arena_fixture_t::allocator_t > fixture_t;

    arena_fixture_t arena_fixture(8 * fixture_t::default_container_size, "TestPerElementMoveAssignOperator");
    fixture_t fixture(arena_fixture.source_allocator);
    fixture_t::cpq_t src_copy(fixture.cpq_src);
    fixture_t::cpq_t dst(arena_fixture.dst_allocator);

    special_member_calls_t move_ctor_called_cpq_size_times = MoveOperationTracker::special_member_calls();
    move_ctor_called_cpq_size_times.move_constructor_called_times += fixture.container_size;
    dst = std::move(fixture.cpq_src);
    ASSERT(move_ctor_called_cpq_size_times == MoveOperationTracker::special_member_calls(), "per element move assignment should move initialize new elements");

    ASSERT(dst == src_copy, "cpq content changed during per element move ?");
}

void TestgMoveAssignOperator(){
    TestStealingMoveAssignOperator();
#if    __TBB_ALLOCATOR_TRAITS_PRESENT
    TestStealingMoveAssignOperatorWithStatefulAllocator();
#endif //__TBB_ALLOCATOR_TRAITS_PRESENT
    TestPerElementMoveAssignOperator();
}

struct ForwardInEmplaceTester {
    int a;
    static bool moveCtorCalled;
    ForwardInEmplaceTester( int a_val ) : a( a_val ) {}
    ForwardInEmplaceTester( ForwardInEmplaceTester&& obj, int a_val ) : a( obj.a ) {
        moveCtorCalled = true;
        obj.a = a_val;
    }
    bool operator<( ForwardInEmplaceTester const& ) const { return true; }
};
bool ForwardInEmplaceTester::moveCtorCalled = false;

struct NoDefaultCtorType {
    size_t value1, value2;
    NoDefaultCtorType( size_t value1_, size_t value2_ ) : value1( value1_ ), value2( value2_ ) {}
    bool operator<(NoDefaultCtorType const &m) const {
        return value1+value2 < m.value1+m.value2;
    }
};

void TestMoveSupportInPushPop() {
    REMARK("Testing Move Support in Push/Pop...");
    size_t &mcct = MoveOperationTracker::move_constructor_called_times;
    size_t &ccct = MoveOperationTracker::copy_constructor_called_times;
    size_t &cact = MoveOperationTracker::copy_assignment_called_times;
    size_t &mact = MoveOperationTracker::move_assignment_called_times;
    mcct = ccct = cact = mact = 0;

    concurrent_priority_queue<MoveOperationTracker> q1;

    ASSERT(mcct == 0, "Value must be zero-initialized");
    ASSERT(ccct == 0, "Value must be zero-initialized");

    q1.push(MoveOperationTracker());
    ASSERT(mcct > 0, "Not working push(T&&)?");
    ASSERT(ccct == 0, "Copying of arg occurred during push(T&&)");

    MoveOperationTracker ob;
    const size_t prev_mcct = mcct;
    q1.push(std::move(ob));
    ASSERT(mcct > prev_mcct, "Not working push(T&&)?");
    ASSERT(ccct == 0, "Copying of arg occurred during push(T&&)");

    ASSERT(cact == 0, "Copy assignment called during push(T&&)");
    const size_t prev_mact = mact;
    q1.try_pop(ob);
    ASSERT(cact == 0, "Copy assignment called during try_pop(T&)");
    ASSERT(mact > prev_mact, "Move assignment was not called during try_pop(T&)");

    REMARK(" works.\n");

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    REMARK("Testing Emplace...");

    concurrent_priority_queue<NoDefaultCtorType> q2;
    q2.emplace(15, 3);
    q2.emplace(2, 35);
    q2.emplace(8, 8);

    NoDefaultCtorType o(0, 0);
    q2.try_pop(o);
    ASSERT(o.value1 == 2 && o.value2 == 35, "Unexpected data popped; possible emplace() failure.");
    q2.try_pop(o);
    ASSERT(o.value1 == 15 && o.value2 == 3, "Unexpected data popped; possible emplace() failure.");
    q2.try_pop(o);
    ASSERT(o.value1 == 8 && o.value2 == 8, "Unexpected data popped; possible emplace() failure.");
    ASSERT(!q2.try_pop(o), "The queue should be empty.");

    //TODO: revise this test
    concurrent_priority_queue<ForwardInEmplaceTester> q3;
    ASSERT( ForwardInEmplaceTester::moveCtorCalled == false, NULL );
    q3.emplace( ForwardInEmplaceTester(5), 2 );
    ASSERT( ForwardInEmplaceTester::moveCtorCalled == true, "Not used std::forward in emplace()?" );
    ForwardInEmplaceTester obj( 0 );
    q3.try_pop( obj );
    ASSERT( obj.a == 5, "Not used std::forward in emplace()?" );
    ASSERT(!q3.try_pop( obj ), "The queue should be empty.");

    REMARK(" works.\n");
#endif /* __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
}
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

void TestCpqOnNThreads( int nThreads ) {
    std::less<int> int_compare;
    my_less data_compare;

    TestConstructorsDestructorsAccessors();
    TestAssignmentClearSwap();
    TestSerialPushPop();

    TestParallelPushPop( nThreads, INT_MAX, INT_MIN, int_compare );
    TestParallelPushPop( nThreads, (unsigned char)CHAR_MAX, (unsigned char)CHAR_MIN, int_compare );
    TestParallelPushPop( nThreads, DATA_MAX, DATA_MIN, data_compare );

    TestFlogger( nThreads, INT_MAX, int_compare );
    TestFlogger( nThreads, (unsigned char)CHAR_MAX, int_compare );
    TestFlogger( nThreads, DATA_MAX, data_compare );
#if __TBB_CPP11_RVALUE_REF_PRESENT
    MoveOperationTracker::copy_assignment_called_times = 0;
    TestFlogger( nThreads, MoveOperationTracker(), std::less<MoveOperationTracker>() );
    ASSERT( MoveOperationTracker::copy_assignment_called_times == 0, "Copy assignment called during try_pop(T&)?" );
#endif

#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    TestExceptions();
#else
    REPORT( "Known issue: exception handling tests are skipped.\n" );
#endif
}

#if __TBB_CPP11_SMART_POINTERS_PRESENT
struct SmartPointersCompare {
    template <typename Type> bool operator() (const std::shared_ptr<Type> &t1, const std::shared_ptr<Type> &t2) {
        return *t1 < *t2;
    }
    template <typename Type> bool operator() (const std::weak_ptr<Type> &t1, const std::weak_ptr<Type> &t2) {
        return *t1.lock().get() < *t2.lock().get();
    }
    template <typename Type> bool operator() (const std::unique_ptr<Type> &t1, const std::unique_ptr<Type> &t2) {
        return *t1 < *t2;
    }
};
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */

#if __TBB_CPP11_RVALUE_REF_PRESENT
// The helper calls copying or moving push operator if an element has copy constructor.
// Otherwise it calls only moving push operator.
template <bool hasCopyCtor>
struct QueuePushHelper {
    template <typename Q, typename T>
    static void push( Q &q, T &&t ) {
        q.push( std::forward<T>(t) );
    }
};
template <>
template <typename Q, typename T>
void QueuePushHelper<false>::push( Q &q, T &&t ) {
    q.push( std::move(t) );
}
#else
template <bool hasCopyCtor>
struct QueuePushHelper {
    template <typename Q, typename T>
    static void push( Q &q, const T &t ) {
        q.push( t );
    }
};
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

template <bool hasCopyCtor, typename Queue>
void Examine(Queue &q1, Queue &q2, const std::vector<typename Queue::value_type> &vecSorted) {
    typedef typename Queue::value_type ValueType;

    ASSERT(!q1.empty() && q1.size() == vecSorted.size(), NULL);

    ValueType elem;

    q2.clear();
    ASSERT(q2.empty() && !q2.size() && !q2.try_pop(elem), NULL);

    typename std::vector<ValueType>::const_reverse_iterator it1;
    for (it1 = vecSorted.rbegin(); q1.try_pop(elem); it1++) {
        ASSERT( Harness::IsEqual()(elem, *it1), NULL );
        if ( std::distance(vecSorted.rbegin(), it1) % 2 )
            QueuePushHelper<hasCopyCtor>::push(q2,elem);
        else
            QueuePushHelper<hasCopyCtor>::push(q2,tbb::internal::move(elem));
    }
    ASSERT(it1 == vecSorted.rend(), NULL);
    ASSERT(q1.empty() && !q1.size(), NULL);
    ASSERT(!q2.empty() && q2.size() == vecSorted.size(), NULL);

    q1.swap(q2);
    ASSERT(q2.empty() && !q2.size(), NULL);
    ASSERT(!q1.empty() && q1.size() == vecSorted.size(), NULL);
    for (it1 = vecSorted.rbegin(); q1.try_pop(elem); it1++) ASSERT(Harness::IsEqual()(elem, *it1), NULL);
    ASSERT(it1 == vecSorted.rend(), NULL);

    typename Queue::allocator_type a = q1.get_allocator();
    ValueType *ptr = a.allocate(1);
    ASSERT(ptr, NULL);
    a.deallocate(ptr, 1);
}

template <typename Queue>
void Examine(const Queue &q, const std::vector<typename Queue::value_type> &vecSorted) {
    Queue q1(q), q2(q);
    Examine</*hasCopyCtor=*/true>( q1, q2, vecSorted );
}

template <typename ValueType, typename Compare>
void TypeTester(const std::vector<ValueType> &vec, Compare comp) {
    typedef tbb::concurrent_priority_queue<ValueType, Compare> Queue;
    typedef tbb::concurrent_priority_queue< ValueType, Compare, debug_allocator<ValueType> > QueueDebugAlloc;
    __TBB_ASSERT(vec.size() >= 5, "Array should have at least 5 elements");

    std::vector<ValueType> vecSorted(vec);
    std::sort( vecSorted.begin(), vecSorted.end(), comp );

    // Construct an empty queue.
    Queue q1;
    q1.assign(vec.begin(), vec.end());
    Examine(q1, vecSorted);
#if __TBB_INITIALIZER_LISTS_PRESENT
    // Constructor from initializer_list.
    Queue q2({ vec[0], vec[1], vec[2] });
    for (typename std::vector<ValueType>::const_iterator it = vec.begin() + 3; it != vec.end(); ++it) q2.push(*it);
    Examine(q2, vecSorted);
    Queue q3;
    q3 = { vec[0], vec[1], vec[2] };
    for (typename std::vector<ValueType>::const_iterator it = vec.begin() + 3; it != vec.end(); ++it) q3.push(*it);
    Examine(q3, vecSorted);
#endif
    // Copying constructor.
    Queue q4(q1);
    Examine(q4, vecSorted);
    // Construct with non-default allocator.
    QueueDebugAlloc q5;
    q5.assign(vec.begin(), vec.end());
    Examine(q5, vecSorted);
    // Copying constructor for vector with different allocator type.
    QueueDebugAlloc q6(q5);
    Examine(q6, vecSorted);
    // Construction with copying iteration range and given allocator instance.
    Queue q7(vec.begin(), vec.end());
    Examine(q7, vecSorted);
    typename QueueDebugAlloc::allocator_type a;
    QueueDebugAlloc q8(a);
    q8.assign(vec.begin(), vec.end());
    Examine(q8, vecSorted);
}

#if __TBB_CPP11_SMART_POINTERS_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT
template <typename T>
void TypeTesterUniquePtr(const std::vector<T> &vec) {
    __TBB_ASSERT(vec.size() >= 5, "Array should have at least 5 elements");

    typedef std::unique_ptr<T> ValueType;
    typedef tbb::concurrent_priority_queue<ValueType, SmartPointersCompare> Queue;
    typedef tbb::concurrent_priority_queue< ValueType, SmartPointersCompare, debug_allocator<ValueType> > QueueDebugAlloc;

    std::vector<ValueType> vecSorted;
    for ( typename std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); ++it ) {
        vecSorted.push_back( ValueType(new T(*it)) );
    }
    std::sort( vecSorted.begin(), vecSorted.end(), SmartPointersCompare() );

    Queue q1, q1Copy;
    QueueDebugAlloc q2, q2Copy;
    for ( typename std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); ++it ) {
        q1.push( ValueType(new T(*it)) );
        q1Copy.push( ValueType(new T(*it)) );
        q2.push( ValueType(new T(*it)) );
        q2Copy.push( ValueType(new T(*it)) );
    }
    Examine</*isCopyCtor=*/false>(q1, q1Copy, vecSorted);
    Examine</*isCopyCtor=*/false>(q2, q2Copy, vecSorted);

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    Queue q3Copy;
    QueueDebugAlloc q4Copy;

    q1.clear();
    q2.clear();
    for ( typename std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); ++it ) {
        q1.emplace( new T(*it) );
        q3Copy.emplace( new T(*it) );
        q2.emplace( new T(*it) );
        q4Copy.emplace( new T(*it) );
    }

    Queue q3( std::move(q1) );
    QueueDebugAlloc q4( std::move(q2) );
    Examine</*isCopyCtor=*/false>(q3, q3Copy, vecSorted);
    Examine</*isCopyCtor=*/false>(q4, q4Copy, vecSorted);
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
}
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT */

template <typename ValueType>
void TypeTester(const std::vector<ValueType> &vec) { TypeTester(vec, std::less<ValueType>()); }

void TestTypes() {
    const int NUMBER = 10;

    Harness::FastRandom rnd(1234);

    std::vector<int> arrInt;
    for (int i = 0; i<NUMBER; ++i) arrInt.push_back(rnd.get());
    std::vector< tbb::atomic<int> > arrTbb;
    for (int i = 0; i<NUMBER; ++i) {
        tbb::atomic<int> a;
        a = rnd.get();
        arrTbb.push_back(a);
    }

    TypeTester(arrInt);
    TypeTester(arrTbb);

#if __TBB_CPP11_SMART_POINTERS_PRESENT
    std::vector< std::shared_ptr<int> > arrShr;
    for (int i = 0; i<NUMBER; ++i) {
        const int rnd_get = rnd.get();
        arrShr.push_back(std::make_shared<int>(rnd_get));
    }
    std::vector< std::weak_ptr<int> > arrWk;
    std::copy(arrShr.begin(), arrShr.end(), std::back_inserter(arrWk));
    TypeTester(arrShr, SmartPointersCompare());
    TypeTester(arrWk, SmartPointersCompare());

#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT
#if __TBB_IS_COPY_CONSTRUCTIBLE_BROKEN
    REPORT( "Known issue: std::is_copy_constructible is broken for move-only types. So the std::unique_ptr test is skipped.\n" );
#else
    TypeTesterUniquePtr(arrInt);
#endif /* __TBB_IS_COPY_CONSTRUCTIBLE_BROKEN */
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT */
#else
    REPORT( "Known issue: C++11 smart pointer tests are skipped.\n" );
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */
}

int TestMain() {
    if (MinThread < 1)
        MinThread = 1;

    TestHelpers();
#if __TBB_INITIALIZER_LISTS_PRESENT
    TestInitList();
#else
    REPORT("Known issue: initializer list tests are skipped.\n");
#endif

    TestTypes();

#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestgMoveConstructor();
    TestgMoveAssignOperator();
    TestMoveSupportInPushPop();
#else
    REPORT("Known issue: move support tests are skipped.\n");
#endif

    for (int p = MinThread; p <= MaxThread; ++p) {
        REMARK("Testing on %d threads.\n", p);
        TestCpqOnNThreads(p);
    }
    return Harness::Done;
}
