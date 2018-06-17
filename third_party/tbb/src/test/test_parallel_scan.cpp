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

#include "tbb/parallel_scan.h"
#include "tbb/blocked_range.h"
#include "harness_assert.h"
#include <vector>

typedef tbb::blocked_range<long> Range;

static volatile bool ScanIsRunning = false;

//! Sum of 0..i with wrap around on overflow.
inline int TriangularSum( int i ) {
    return i&1 ? ((i>>1)+1)*i : (i>>1)*(i+1);
}

#include "harness.h"

//! Verify that sum is init plus sum of integers in closed interval [0..finish_index].
/** line should be the source line of the caller */
void VerifySum( int init, long finish_index, int sum, int line ) {
    int expected = init + TriangularSum(finish_index);
    if (expected != sum) {
        REPORT("line %d: sum[0..%ld] should be = %d, but was computed as %d\n",
            line, finish_index, expected, sum);
        abort();
    }
}

const int MAXN = 2000;

enum AddendFlag {
    UNUSED=0,
    USED_NONFINAL=1,
    USED_FINAL=2
};

//! Array recording how each addend was used.
/** 'unsigned char' instead of AddendFlag for sake of compactness. */
static unsigned char AddendHistory[MAXN];

//! Set to 1 for debugging output
#define PRINT_DEBUG 0

#include "tbb/atomic.h"
#if PRINT_DEBUG
#include <stdio.h>
#include "harness_report.h"
tbb::atomic<long> NextBodyId;
#endif /* PRINT_DEBUG */

struct BodyId {
#if PRINT_DEBUG
    const int id;
    BodyId() : id(NextBodyId++) {}
#endif /* PRINT_DEBUG */
};

tbb::atomic<long> NumberOfLiveStorage;

static void Snooze( bool scan_should_be_running ) {
    ASSERT( ScanIsRunning==scan_should_be_running, NULL );
}

template<typename T>
struct Storage {
    T my_total;
    Range my_range;
    Storage(T init) :
        my_total(init), my_range(-1, -1, 1) {
        ++NumberOfLiveStorage;
    }
    ~Storage() {
        --NumberOfLiveStorage;
    }
    Storage(const Storage& strg) :
        my_total(strg.my_total), my_range(strg.my_range) {
        ++NumberOfLiveStorage;
    }
    Storage & operator=(const Storage& strg) {
        my_total = strg.my_total;
        my_range = strg.my_range;
        return *this;
    }
};

template<typename T>
void JoinStorages(const Storage<T>& left, Storage<T>& right) {
    Snooze(true);
    ASSERT(ScanIsRunning, NULL);
    ASSERT(left.my_range.end() == right.my_range.begin(), NULL);
    right.my_total += left.my_total;
    right.my_range = Range(left.my_range.begin(), right.my_range.end(), 1);
    ASSERT(ScanIsRunning, NULL);
    Snooze(true);
    ASSERT(ScanIsRunning, NULL);
}

template<typename T>
void Scan(const Range & r, bool is_final, Storage<T> & storage, std::vector<T> & sum, const std::vector<T> & addend) {
    ASSERT(!is_final || (storage.my_range.begin() == 0 && storage.my_range.end() == r.begin()) || (storage.my_range.empty() && r.begin() == 0), NULL);
    for (long i = r.begin(); i < r.end(); ++i) {
        storage.my_total += addend[i];
        if (is_final) {
            ASSERT(AddendHistory[i] < USED_FINAL, "addend used 'finally' twice?");
            AddendHistory[i] |= USED_FINAL;
            sum[i] = storage.my_total;
            VerifySum(42, i, int(sum[i]), __LINE__);
        }
        else {
            ASSERT(AddendHistory[i] == UNUSED, "addend used too many times");
            AddendHistory[i] |= USED_NONFINAL;
        }
    }
    if (storage.my_range.empty())
        storage.my_range = r;
    else
        storage.my_range = Range(storage.my_range.begin(), r.end(), 1);
    Snooze(true);
}

template<typename T>
Storage<T> ScanWithInit(const Range & r, T init, bool is_final, Storage<T> & storage, std::vector<T> & sum, const std::vector<T> & addend) {
    if (r.begin() == 0)
        storage.my_total = init;
    Scan(r, is_final, storage, sum, addend);
    return storage;
}

template<typename T>
class Accumulator: BodyId {
    const  std::vector<T> &my_array;
    std::vector<T> & my_sum;
    Storage<T> storage;
    enum state_type {
        full,       // Accumulator has sufficient information for final scan,
                    // i.e. has seen all iterations to its left.
                    // It's either the original Accumulator provided by the user
                    // or a Accumulator constructed by a splitting constructor *and* subsequently
                    // subjected to a reverse_join with a full accumulator.

        partial,    // Accumulator has only enough information for pre_scan.
                    // i.e. has not seen all iterations to its left.
                    // It's an Accumulator created by a splitting constructor that
                    // has not yet been subjected to a reverse_join with a full accumulator.

        summary,    // Accumulator has summary of iterations processed, but not necessarily
                    // the information required for a final_scan or pre_scan.
                    // It's the result of "assign".

        trash       // Accumulator with possibly no useful information.
                    // It was the source for "assign".

    };
    mutable state_type my_state;
    //! Equals this while object is fully constructed, NULL otherwise.
    /** Used to detect premature destruction and accidental bitwise copy. */
    Accumulator* self;
    Accumulator& operator= (const Accumulator& other);
public:
    Accumulator( T init, const std::vector<T> & array, std::vector<T> & sum ) :
        my_array(array), my_sum(sum), storage(init), my_state(full)
    {
        // Set self as last action of constructor, to indicate that object is fully constructed.
        self = this;
    }
#if PRINT_DEBUG
    void print() const {
        REPORT("%d [%ld..%ld)\n", id, storage.my_range.begin(), storage.my_range.end() );
    }
#endif /* PRINT_DEBUG */
    ~Accumulator() {
#if PRINT_DEBUG
        REPORT("%d [%ld..%ld) destroyed\n",id, storage.my_range.begin(), storage.my_range.end() );
#endif /* PRINT_DEBUG */
        // Clear self as first action of destructor, to indicate that object is not fully constructed.
        self = 0;
    }
    Accumulator( Accumulator& a, tbb::split ) :
        my_array(a.my_array), my_sum(a.my_sum), storage(0), my_state(partial)
    {
        ASSERT(a.my_state==full || a.my_state==partial, NULL);
#if PRINT_DEBUG
        REPORT("%d forked from %d\n",id,a.id);
#endif /* PRINT_DEBUG */
        Snooze(true);
        // Set self as last action of constructor, to indicate that object is fully constructed.
        self = this;
    }
    template<typename Tag>
    void operator()( const Range& r, Tag /*tag*/ ) {
        ASSERT( Tag::is_final_scan() ? my_state==full : my_state==partial, NULL );
#if PRINT_DEBUG
        if(storage.my_range.empty() )
            REPORT("%d computing %s [%ld..%ld)\n",id,Tag::is_final_scan()?"final":"lookahead",r.begin(),r.end() );
        else
            REPORT("%d computing %s [%ld..%ld) [%ld..%ld)\n",id,Tag::is_final_scan()?"final":"lookahead", storage.my_range.begin(), storage.my_range.end(),r.begin(),r.end());
#endif /* PRINT_DEBUG */
        Scan(r, Tag::is_final_scan(), storage, my_sum, my_array);
        ASSERT( self==this, "this Accumulator corrupted or prematurely destroyed" );
    }
    void reverse_join( const Accumulator& left_body) {
#if PRINT_DEBUG
        REPORT("reverse join %d [%ld..%ld) %d [%ld..%ld)\n",
               left_body.id, left_body.storage.my_range.begin(), left_body.storage.my_range.end(),
               id, storage.my_range.begin(), storage.my_range.end());
#endif /* PRINT_DEBUG */
        const Storage<T> & left = left_body.storage;
        Storage<T> & right = storage;
        ASSERT(my_state==partial, NULL );
        ASSERT(left_body.my_state==full || left_body.my_state==partial, NULL );

        JoinStorages(left, right);

        ASSERT(left_body.self==&left_body, NULL );
        my_state = left_body.my_state;
    }
    void assign( const Accumulator& other ) {
        ASSERT(other.my_state==full, NULL);
        ASSERT(my_state==full, NULL);
        storage.my_total = other.storage.my_total;
        storage.my_range = other.storage.my_range;
        ASSERT( self==this, NULL );
        ASSERT( other.self==&other, "other Accumulator corrupted or prematurely destroyed" );
        my_state = summary;
        other.my_state = trash;
    }
    T get_total() {
        return storage.my_total;
    }
};

#include "tbb/tick_count.h"

template<typename T, typename Scan, typename ReverseJoin>
T ParallelScanFunctionalInvoker(const Range& range, T idx, const Scan& scan, const ReverseJoin& reverse_join, int mode) {
    switch (mode%3) {
    case 0:
        return tbb::parallel_scan(range, idx, scan, reverse_join);
        break;
    case 1:
        return tbb::parallel_scan(range, idx, scan, reverse_join, tbb::simple_partitioner());
        break;
    default:
        return tbb::parallel_scan(range, idx, scan, reverse_join, tbb::auto_partitioner());
    }
}

template<typename T>
class ScanBody {
    const std::vector<T> &my_addend;
    std::vector<T> &my_sum;
    const T my_init;
    ScanBody& operator= (const ScanBody&);
public:
    ScanBody(T init, const std::vector<T> &addend, std::vector<T> &sum) :my_addend(addend), my_sum(sum), my_init(init) {}
    template<typename Tag>
    Storage<T> operator()(const Range& r, Storage<T> storage, Tag) const {
        return ScanWithInit(r, my_init, Tag::is_final_scan(), storage, my_sum, my_addend);
    }
};

template<typename T>
class JoinBody {
public:
    Storage<T> operator()(const Storage<T>& left, Storage<T>& right) const {
        JoinStorages(left, right);
        return right;
    }
};

template<typename T>
T ParallelScanTemplateFunctor(Range range, T init, const std::vector<T> &addend, std::vector<T> &sum, int mode) {
    for (long i = 0; i<MAXN; ++i) {
        AddendHistory[i] = UNUSED;
    }
    ScanIsRunning = true;
    ScanBody<T> sb(init, addend, sum);
    JoinBody<T> jb;
    Storage<T> res = ParallelScanFunctionalInvoker(range, Storage<T>(0), sb, jb, mode);
    ScanIsRunning = false;
    if (range.empty())
        res.my_total = init;
    return res.my_total;
}

#if __TBB_CPP11_LAMBDAS_PRESENT
template<typename T>
T ParallelScanLambda(Range range, T init, const std::vector<T> &addend, std::vector<T> &sum, int mode) {
    for (long i = 0; i<MAXN; ++i) {
        AddendHistory[i] = UNUSED;
    }
    ScanIsRunning = true;
    Storage<T> res = ParallelScanFunctionalInvoker(range, Storage<T>(0),
        [&addend, &sum, init](const Range& r, Storage<T> storage, bool is_final_scan /*tag*/) -> Storage<T> {
            return ScanWithInit(r, init, is_final_scan, storage, sum, addend);
        },
        [](const Storage<T>& left, Storage<T>& right) -> Storage<T> {
            JoinStorages(left, right);
            return right;
        },
        mode);
    ScanIsRunning = false;
    if (range.empty())
        res.my_total = init;
    return res.my_total;
}

#if __TBB_CPP14_GENERIC_LAMBDAS_PRESENT
template<typename T>
T ParallelScanGenericLambda(Range range, T init, const std::vector<T> &addend, std::vector<T> &sum, int mode) {
    for (long i = 0; i<MAXN; ++i) {
        AddendHistory[i] = UNUSED;
    }
    ScanIsRunning = true;
    Storage<T> res = ParallelScanFunctionalInvoker(range, Storage<T>(0),
        [&addend, &sum, init](const Range& rng, Storage<T> storage, auto scan_tag) {
            return ScanWithInit(rng, init, scan_tag.is_final_scan(), storage, sum, addend);
        },
        [](const Storage<T>& left, Storage<T>& right) {
            JoinStorages(left, right);
            return right;
        },
        mode);
    ScanIsRunning = false;
    if (range.empty())
        res.my_total = init;
    return res.my_total;
}
#endif/* GENERIC_LAMBDAS */
#endif/* LAMBDAS */

void TestAccumulator( int mode, int nthread ) {
    typedef int T;
    std::vector<T> addend(MAXN);
    std::vector<T> sum(MAXN);
    for( long n=0; n<=MAXN; ++n ) {
        for( long i=0; i<MAXN; ++i ) {
            addend[i] = -1;
            sum[i] = -2;
            AddendHistory[i] = UNUSED;
        }
        for( long i=0; i<n; ++i )
            addend[i] = i;

        Accumulator<T> acc( 42, addend, sum );
        tbb::tick_count t0 = tbb::tick_count::now();
#if PRINT_DEBUG
        REPORT("--------- mode=%d range=[0..%ld)\n",mode,n);
#endif /* PRINT_DEBUG */
        ScanIsRunning = true;

        switch (mode) {
            case 0:
                tbb::parallel_scan( Range( 0, n, 1 ), acc );
            break;
            case 1:
                tbb::parallel_scan( Range( 0, n, 1 ), acc, tbb::simple_partitioner() );
            break;
            case 2:
                tbb::parallel_scan( Range( 0, n, 1 ), acc, tbb::auto_partitioner() );
            break;
        }

        ScanIsRunning = false;
#if PRINT_DEBUG
        REPORT("=========\n");
#endif /* PRINT_DEBUG */
        Snooze(false);
        tbb::tick_count t1 = tbb::tick_count::now();
        long used_once_count = 0;
        for( long i=0; i<n; ++i )
            if( !(AddendHistory[i]&USED_FINAL) ) {
                REPORT("failed to use addend[%ld] %s\n",i,AddendHistory[i]&USED_NONFINAL?"(but used nonfinal)":"");
            }
        for( long i=0; i<n; ++i ) {
            VerifySum( 42, i, sum[i], __LINE__ );
            used_once_count += AddendHistory[i]==USED_FINAL;
        }
        if( n )
            ASSERT( acc.get_total()==sum[n-1], NULL );
        else
            ASSERT( acc.get_total()==42, NULL );
        REMARK("time [n=%ld] = %g\tused_once%% = %g\tnthread=%d\n",n,(t1-t0).seconds(), n==0 ? 0 : 100.0*used_once_count/n,nthread);


       std::vector<T> sum_tmplt(MAXN);
        for (long i = 0; i<MAXN; ++i)
            sum_tmplt[i] = -2;
        T total_tmplt = ParallelScanTemplateFunctor(Range(0, n, 1), 42, addend, sum_tmplt, mode);

        ASSERT(acc.get_total() == total_tmplt, "Parallel prefix sum with lambda interface is not equal to body interface");
        ASSERT(sum == sum_tmplt, "Parallel prefix vector with lambda interface is not equal to body interface");

#if __TBB_CPP11_LAMBDAS_PRESENT
        std::vector<T> sum_lambda(MAXN);
        for (long i = 0; i<MAXN; ++i)
            sum_lambda[i] = -2;
        T total_lambda = ParallelScanLambda(Range(0, n, 1), 42, addend, sum_lambda, mode);

        ASSERT(acc.get_total() == total_lambda, "Parallel prefix sum with lambda interface is not equal to body interface");
        ASSERT(sum == sum_lambda, "Parallel prefix vector with lambda interface is not equal to body interface");

#if __TBB_CPP14_GENERIC_LAMBDAS_PRESENT
        std::vector<T> sum_generic_lambda(MAXN);
        for (long i = 0; i<MAXN; ++i)
            sum_generic_lambda[i] = -2;
        T total_generic_lambda = ParallelScanGenericLambda(Range(0, n, 1), 42, addend, sum_generic_lambda, mode);

        ASSERT(acc.get_total() == total_generic_lambda, "Parallel prefix sum with lambda (generic) interface is not equal to body interface");
        ASSERT(sum == sum_generic_lambda, "Parallel prefix vector with lambda (generic) interface is not equal to body interface");

#endif /* GENERIC_LAMBDAS */
#endif /* LAMBDAS */
    }
}

static void TestScanTags() {
    ASSERT( tbb::pre_scan_tag::is_final_scan()==false, NULL );
    ASSERT( tbb::final_scan_tag::is_final_scan()==true, NULL );
    ASSERT( tbb::pre_scan_tag() == false, NULL );
    ASSERT( tbb::final_scan_tag() == true, NULL );
}

#include "tbb/task_scheduler_init.h"
#include "harness_cpu.h"

int TestMain () {
    TestScanTags();
    for( int p=MinThread; p<=MaxThread; ++p ) {
        for (int mode = 0; mode < 3; mode++) {
            tbb::task_scheduler_init init(p);
            NumberOfLiveStorage = 0;
            TestAccumulator(mode, p);
            // Test that all workers sleep when no work
            TestCPUUserTime(p);

            // Checking has to be done late, because when parallel_scan makes copies of
            // the user's "Body", the copies might be destroyed slightly after parallel_scan
            // returns.
            ASSERT( NumberOfLiveStorage==0, NULL );
        }
    }
    return Harness::Done;
}
