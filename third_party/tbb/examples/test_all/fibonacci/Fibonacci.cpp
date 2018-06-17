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

/* Example program that computes Fibonacci numbers in different ways.
   Arguments are: [ Number [Threads [Repeats]]]
   The defaults are Number=500 Threads=1:4 Repeats=1.

   The point of this program is to check that the library is working properly.
   Most of the computations are deliberately silly and not expected to
   show any speedup on multiprocessors.
*/

// enable assertions
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <utility>
#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "tbb/blocked_range.h"
#include "tbb/concurrent_vector.h"
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_hash_map.h"
#include "tbb/parallel_while.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_scan.h"
#include "tbb/pipeline.h"
#include "tbb/atomic.h"
#include "tbb/mutex.h"
#include "tbb/spin_mutex.h"
#include "tbb/queuing_mutex.h"
#include "tbb/tbb_thread.h"

using namespace std;
using namespace tbb;

//! type used for Fibonacci number computations
typedef long long value;

//! Matrix 2x2 class
struct Matrix2x2
{
    //! Array of values
    value v[2][2];
    Matrix2x2() {}
    Matrix2x2(value v00, value v01, value v10, value v11) {
        v[0][0] = v00; v[0][1] = v01; v[1][0] = v10; v[1][1] = v11;
    }
    Matrix2x2 operator * (const Matrix2x2 &to) const; //< Multiply two Matrices
};
//! Identity matrix
static const Matrix2x2 MatrixIdentity(1, 0, 0, 1);
//! Default matrix to multiply
static const Matrix2x2 Matrix1110(1, 1, 1, 0);
//! Raw arrays matrices multiply
void Matrix2x2Multiply(const value a[2][2], const value b[2][2], value c[2][2]);

/////////////////////// Serial methods ////////////////////////

//! Plain serial sum
value SerialFib(int n)
{
    if(n < 2)
        return n;
    value a = 0, b = 1, sum; int i;
    for( i = 2; i <= n; i++ )
    {   // n is really index of Fibonacci number
        sum = a + b; a = b; b = sum;
    }
    return sum;
}
//! Serial n-1 matrices multiplication
value SerialMatrixFib(int n)
{
    value c[2][2], a[2][2] = {{1, 1}, {1, 0}}, b[2][2] = {{1, 1}, {1, 0}}; int i;
    for(i = 2; i < n; i++)
    {   // Using condition to prevent copying of values
        if(i & 1) Matrix2x2Multiply(a, c, b);
        else      Matrix2x2Multiply(a, b, c);
    }
    return (i & 1) ? c[0][0] : b[0][0]; // get result from upper left cell
}
//! Recursive summing. Just for complete list of serial algorithms, not used
value SerialRecursiveFib(int n)
{
    value result;
    if(n < 2)
        result = n;
    else
        result = SerialRecursiveFib(n - 1) + SerialRecursiveFib(n - 2);
    return result;
}
//! Introducing of queue method in serial
value SerialQueueFib(int n)
{
    concurrent_queue<Matrix2x2> Q;
    for(int i = 1; i < n; i++)
        Q.push(Matrix1110);
    Matrix2x2 A, B;
    while(true) {
        while( !Q.try_pop(A) ) this_tbb_thread::yield();
        if(Q.empty()) break;
        while( !Q.try_pop(B) ) this_tbb_thread::yield();
        Q.push(A * B);
    }
    return A.v[0][0];
}
//! Trying to use concurrent_vector
value SerialVectorFib(int n)
{
    concurrent_vector<value> A;
    A.grow_by(2);
    A[0] = 0; A[1] = 1;
    for( int i = 2; i <= n; i++)
    {
        A.grow_to_at_least(i+1);
        A[i] = A[i-1] + A[i-2];
    }
    return A[n];
}

///////////////////// Parallel methods ////////////////////////

// *** Serial shared by mutexes *** //

//! Shared glabals
value SharedA = 0, SharedB = 1; int SharedI = 1, SharedN;

//! Template task class which computes Fibonacci numbers with shared globals
template<typename M>
class SharedSerialFibBody {
    M &mutex;
public:
    SharedSerialFibBody( M &m ) : mutex( m ) {}
    //! main loop
    void operator()( const blocked_range<int>& range ) const {
        for(;;) {
            typename M::scoped_lock lock( mutex );
            if(SharedI >= SharedN) break;
            value sum = SharedA + SharedB;
            SharedA = SharedB; SharedB = sum;
            ++SharedI;
        }
    }
};

//! Root function
template<class M>
value SharedSerialFib(int n)
{
    SharedA = 0; SharedB = 1; SharedI = 1; SharedN = n; M mutex;
    parallel_for( blocked_range<int>(0,4,1), SharedSerialFibBody<M>( mutex ) );
    return SharedB;
}

// *** Serial shared by concurrent hash map *** //

//! Hash comparer
struct IntHashCompare {
    bool equal( const int j, const int k ) const { return j == k; }
    unsigned long hash( const int k ) const { return (unsigned long)k; }   
};
//! NumbersTable type based on concurrent_hash_map
typedef concurrent_hash_map<int, value, IntHashCompare> NumbersTable;
//! task for serial method using shared concurrent_hash_map
class ConcurrentHashSerialFibTask: public task {
    NumbersTable &Fib;
    int my_n;
public:
    //! constructor
    ConcurrentHashSerialFibTask( NumbersTable &cht, int n ) : Fib(cht), my_n(n) { }
    //! executing task
    task* execute() /*override*/ {
        for( int i = 2; i <= my_n; ++i ) { // there is no difference in to recycle or to make loop
            NumbersTable::const_accessor f1, f2; // same as iterators
            if( !Fib.find(f1, i-1) || !Fib.find(f2, i-2) ) {
                // Something is seriously wrong, because i-1 and i-2 must have been inserted 
                // earlier by this thread or another thread.
                assert(0);
            }
            value sum = f1->second + f2->second;
            NumbersTable::const_accessor fsum;
            Fib.insert(fsum, make_pair(i, sum)); // inserting
            assert( fsum->second == sum ); // check value
        }
        return 0;
    }
};

//! Root function
value ConcurrentHashSerialFib(int n)
{
    NumbersTable Fib; 
    bool okay;
    okay = Fib.insert( make_pair(0, 0) ); assert(okay); // assign initial values
    okay = Fib.insert( make_pair(1, 1) ); assert(okay);

    task_list list;
    // allocate tasks
    list.push_back(*new(task::allocate_root()) ConcurrentHashSerialFibTask(Fib, n));
    list.push_back(*new(task::allocate_root()) ConcurrentHashSerialFibTask(Fib, n));
    task::spawn_root_and_wait(list);
    NumbersTable::const_accessor fresult;
    okay = Fib.find( fresult, n );
    assert(okay);
    return fresult->second;
}

// *** Queue with parallel_for and parallel_while *** //

//! Stream of matrices
struct QueueStream {
    volatile bool producer_is_done;
    concurrent_queue<Matrix2x2> Queue;
    //! Get pair of matricies if present
    bool pop_if_present( pair<Matrix2x2, Matrix2x2> &mm ) {
        // get first matrix if present
        if(!Queue.try_pop(mm.first)) return false;
        // get second matrix if present
        if(!Queue.try_pop(mm.second)) {
            // if not, then push back first matrix
            Queue.push(mm.first); return false;
        }
        return true;
    }
};

//! Functor for parallel_for which fills the queue
struct parallel_forFibBody { 
    QueueStream &my_stream;
    //! fill functor arguments
    parallel_forFibBody(QueueStream &s) : my_stream(s) { }
    //! iterate thorough range
    void operator()( const blocked_range<int> &range ) const {
        int i_end = range.end();
        for( int i = range.begin(); i != i_end; ++i ) {
            my_stream.Queue.push( Matrix1110 ); // push initial matrix
        }
    }
};
//! Functor for parallel_while which process the queue
class parallel_whileFibBody
{
    QueueStream &my_stream;
    parallel_while<parallel_whileFibBody> &my_while;
public:
    typedef pair<Matrix2x2, Matrix2x2> argument_type;
    //! fill functor arguments
    parallel_whileFibBody(parallel_while<parallel_whileFibBody> &w, QueueStream &s)
        : my_while(w), my_stream(s) { }
    //! process pair of matrices
    void operator() (argument_type mm) const {
        mm.first = mm.first * mm.second;
        // note: it can run concurrently with QueueStream::pop_if_present()
        if(my_stream.Queue.try_pop(mm.second))
             my_while.add( mm ); // now, two matrices available. Add next iteration.
        else my_stream.Queue.push( mm.first ); // or push back calculated value if queue is empty
    }
};

//! Parallel queue's filling task
struct QueueInsertTask: public task {
    QueueStream &my_stream;
    int my_n;
    //! fill task arguments
    QueueInsertTask( int n, QueueStream &s ) : my_n(n), my_stream(s) { }
    //! executing task
    task* execute() /*override*/ {
        // Execute of parallel pushing of n-1 initial matrices
        parallel_for( blocked_range<int>( 1, my_n, 10 ), parallel_forFibBody(my_stream) ); 
        my_stream.producer_is_done = true; 
        return 0;
    }
};
//! Parallel queue's processing task
struct QueueProcessTask: public task {
    QueueStream &my_stream;
    //! fill task argument
    QueueProcessTask( QueueStream &s ) : my_stream(s) { }
    //! executing task
    task* execute() /*override*/ {
        while( !my_stream.producer_is_done || my_stream.Queue.unsafe_size()>1 ) {
            parallel_while<parallel_whileFibBody> w; // run while loop in parallel
            w.run( my_stream, parallel_whileFibBody( w, my_stream ) );
        }
        return 0;
    }
};
//! Root function
value ParallelQueueFib(int n)
{
    QueueStream stream;
    stream.producer_is_done = false;
    task_list list;
    list.push_back(*new(task::allocate_root()) QueueInsertTask( n, stream ));
    list.push_back(*new(task::allocate_root()) QueueProcessTask( stream ));
    // If there is only a single thread, the first task in the list runs to completion
    // before the second task in the list starts.
    task::spawn_root_and_wait(list);
    assert(stream.Queue.unsafe_size() == 1); // it is easy to lose some work
    Matrix2x2 M; 
    bool result = stream.Queue.try_pop( M ); // get last matrix
    assert( result );
    return M.v[0][0]; // and result number
}

// *** Queue with pipeline *** //

//! filter to fills queue
class InputFilter: public filter {
    tbb::atomic<int> N; //< index of Fibonacci number minus 1
public:
    concurrent_queue<Matrix2x2> Queue;
    //! fill filter arguments
    InputFilter( int n ) : filter(false /*is not serial*/) { N = n; }
    //! executing filter
    void* operator()(void*) /*override*/ {
        int n = --N;
        if(n <= 0) return 0;
        Queue.push( Matrix1110 );
        return &Queue;
    }
};
//! filter to process queue
class MultiplyFilter: public filter {
public:
    MultiplyFilter( ) : filter(false /*is not serial*/) { }
    //! executing filter
    void* operator()(void*p) /*override*/ {
        concurrent_queue<Matrix2x2> &Queue = *static_cast<concurrent_queue<Matrix2x2> *>(p);
        Matrix2x2 m1, m2;
        // get two elements
        while( !Queue.try_pop( m1 ) ) this_tbb_thread::yield(); 
        while( !Queue.try_pop( m2 ) ) this_tbb_thread::yield();
        m1 = m1 * m2; // process them
        Queue.push( m1 ); // and push back
        return this; // just nothing
    }
};
//! Root function
value ParallelPipeFib(int n)
{
    InputFilter input( n-1 );
    MultiplyFilter process;
    // Create the pipeline
    pipeline pipeline;
    // add filters
    pipeline.add_filter( input ); // first
    pipeline.add_filter( process ); // second

    input.Queue.push( Matrix1110 );
    // Run the pipeline
    pipeline.run( n ); // must be larger then max threads number
    pipeline.clear(); // do not forget clear the pipeline

    assert( input.Queue.unsafe_size()==1 );
    Matrix2x2 M; 
    bool result = input.Queue.try_pop( M ); // get last element
    assert( result );
    return M.v[0][0]; // get value
}

// *** parallel_reduce *** //

//! Functor for parallel_reduce
struct parallel_reduceFibBody {
    Matrix2x2 sum;
    int splitted;  //< flag to make one less operation for splitted bodies
    //! Constructor fills sum with initial matrix
    parallel_reduceFibBody() : sum( Matrix1110 ), splitted(0) { }
    //! Splitting constructor
    parallel_reduceFibBody( parallel_reduceFibBody& other, split ) : sum( Matrix1110 ), splitted(1/*note that it is splitted*/) {}
    //! Join point
    void join( parallel_reduceFibBody &s ) {
        sum = sum * s.sum;
    }
    //! Process multiplications
    void operator()( const blocked_range<int> &r ) {
        for( int k = r.begin() + splitted; k < r.end(); ++k )
            sum = sum * Matrix1110;
        splitted = 0; // reset flag, because this method can be reused for next range
    }
};
//! Root function
value parallel_reduceFib(int n)
{
    parallel_reduceFibBody b;
    parallel_reduce(blocked_range<int>(2, n, 3), b); // do parallel reduce on range [2, n) for b
    return b.sum.v[0][0];
}

// *** parallel_scan *** //

//! Functor for parallel_scan
struct parallel_scanFibBody {
    /** Though parallel_scan is usually used to accumulate running sums,
        it can be used to accumulate running products too. */
    Matrix2x2 product;
    /** Pointer to output sequence */
    value* const output;
    //! Constructor sets product to identity matrix
    parallel_scanFibBody(value* output_) : product( MatrixIdentity ), output(output_) {}
    //! Splitting constructor
    parallel_scanFibBody( parallel_scanFibBody &b, split) : product( MatrixIdentity ), output(b.output) {}
    //! Method for merging summary information from a, which was split off from *this, into *this.
    void reverse_join( parallel_scanFibBody &a ) {
        // When using non-commutative reduction operation, reverse_join
        // should put argument "a" on the left side of the operation.
        // The reversal from the argument order is why the method is
        // called "reverse_join" instead of "join".
        product = a.product * product;
    }
    //! Method for assigning final result back to original body.
    void assign( parallel_scanFibBody &b ) {
        product = b.product;
    }
    //! Compute matrix running product.
    /** Tag indicates whether is is the final scan over the range, or
        just a helper "prescan" that is computing a partial reduction. */
    template<typename Tag>
    void operator()( const blocked_range<int> &r, Tag tag) {
        for( int k = r.begin(); k < r.end(); ++k ) {
            // Code performs an "exclusive" scan, which outputs a value *before* updating the product.
            // For an "inclusive" scan, output the value after the update.
            if( tag.is_final_scan() )
                output[k] = product.v[0][1];
            product = product * Matrix1110;
        }
    }
};
//! Root function
value parallel_scanFib(int n)
{
    value* output = new value[n];
    parallel_scanFibBody b(output);
    parallel_scan(blocked_range<int>(0, n, 3), b);
    // output[0..n-1] now contains the Fibonacci sequence (modulo integer wrap-around).
    // Check the last two values for correctness.
    assert( n<2 || output[n-2]+output[n-1]==b.product.v[0][1] );
    delete[] output;
    return b.product.v[0][1];
}

// *** Raw tasks *** //

//! task class which computes Fibonacci numbers by Lucas formula
struct FibTask: public task {
    const int n;
    value& sum;
    value x, y;
    bool second_phase; //< flag of continuation
    // task arguments
    FibTask( int n_, value& sum_ ) : 
        n(n_), sum(sum_), second_phase(false)
    {}
    //! Execute task
    task* execute() /*override*/ {
        // Using Lucas' formula here
        if( second_phase ) { // children finished
            sum = n&1 ? x*x + y*y : x*x - y*y;
            return NULL;
        }
        if( n <= 2 ) {
            sum = n!=0;
            return NULL;
        } else {
            recycle_as_continuation();  // repeat this task when children finish
            second_phase = true; // mark second phase
            FibTask& a = *new( allocate_child() ) FibTask( n/2 + 1, x );
            FibTask& b = *new( allocate_child() ) FibTask( n/2 - 1 + (n&1), y );
            set_ref_count(2);
            spawn( a );
            return &b;
        }
    }
};
//! Root function
value ParallelTaskFib(int n) { 
    value sum;
    FibTask& a = *new(task::allocate_root()) FibTask(n, sum);
    task::spawn_root_and_wait(a);
    return sum;
}

/////////////////////////// Main ////////////////////////////////////////////////////

//! A closed range of int.
struct IntRange {
    int low;
    int high;
    void set_from_string( const char* s );
    IntRange( int low_, int high_ ) : low(low_), high(high_) {}
};

void IntRange::set_from_string( const char* s ) {
    char* end;
    high = low = strtol(s,&end,0);
    switch( *end ) {
    case ':': 
        high = strtol(end+1,0,0); 
        break;
    case '\0':
        break;
    default:
        printf("unexpected character = %c\n",*end);
    }
}

//! Tick count for start
static tick_count t0;

//! Verbose output flag
static bool Verbose = false;

typedef value (*MeasureFunc)(int);
//! Measure ticks count in loop [2..n]
value Measure(const char *name, MeasureFunc func, int n)
{
    value result;
    if(Verbose) printf("%s",name);
    t0 = tick_count::now();
    for(int number = 2; number <= n; number++)
        result = func(number);
    if(Verbose) printf("\t- in %f msec\n", (tick_count::now() - t0).seconds()*1000);
    return result;
}

//! program entry
int main(int argc, char* argv[])
{
    if(argc>1) Verbose = true;
    int NumbersCount = argc>1 ? strtol(argv[1],0,0) : 500;
    IntRange NThread(1,4);// Number of threads to use.
    if(argc>2) NThread.set_from_string(argv[2]);
    unsigned long ntrial = argc>3? (unsigned long)strtoul(argv[3],0,0) : 1;
    value result, sum;

    if(Verbose) printf("Fibonacci numbers example. Generating %d numbers..\n",  NumbersCount);

    result = Measure("Serial loop", SerialFib, NumbersCount);
    sum = Measure("Serial matrix", SerialMatrixFib, NumbersCount); assert(result == sum);
    sum = Measure("Serial vector", SerialVectorFib, NumbersCount); assert(result == sum);
    sum = Measure("Serial queue", SerialQueueFib, NumbersCount); assert(result == sum);
    // now in parallel
    for( unsigned long i=0; i<ntrial; ++i ) {
        for(int threads = NThread.low; threads <= NThread.high; threads *= 2)
        {
            task_scheduler_init scheduler_init(threads);
            if(Verbose) printf("\nThreads number is %d\n", threads);

            sum = Measure("Shared serial (mutex)\t", SharedSerialFib<mutex>, NumbersCount); assert(result == sum);
            sum = Measure("Shared serial (spin_mutex)", SharedSerialFib<spin_mutex>, NumbersCount); assert(result == sum);
            sum = Measure("Shared serial (queuing_mutex)", SharedSerialFib<queuing_mutex>, NumbersCount); assert(result == sum);
            sum = Measure("Shared serial (Conc.HashTable)", ConcurrentHashSerialFib, NumbersCount); assert(result == sum);
            sum = Measure("Parallel while+for/queue", ParallelQueueFib, NumbersCount); assert(result == sum);
            sum = Measure("Parallel pipe/queue\t", ParallelPipeFib, NumbersCount); assert(result == sum);
            sum = Measure("Parallel reduce\t\t", parallel_reduceFib, NumbersCount); assert(result == sum);
            sum = Measure("Parallel scan\t\t", parallel_scanFib, NumbersCount); assert(result == sum);
            sum = Measure("Parallel tasks\t\t", ParallelTaskFib, NumbersCount); assert(result == sum);
        }

    #ifdef __GNUC__
        if(Verbose) printf("Fibonacci number #%d modulo 2^64 is %lld\n\n", NumbersCount, result);
    #else
        if(Verbose) printf("Fibonacci number #%d modulo 2^64 is %I64d\n\n", NumbersCount, result);
    #endif
    }
    if(!Verbose) printf("TEST PASSED\n");
    return 0;
}

// Utils

void Matrix2x2Multiply(const value a[2][2], const value b[2][2], value c[2][2])
{
    for( int i = 0; i <= 1; i++)
        for( int j = 0; j <= 1; j++)
            c[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j];
}

Matrix2x2 Matrix2x2::operator *(const Matrix2x2 &to) const
{
    Matrix2x2 result;
    Matrix2x2Multiply(v, to.v, result.v);
    return result;
}
