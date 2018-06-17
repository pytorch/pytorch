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

#define HARNESS_DEFAULT_MIN_THREADS 1
#define HARNESS_DEFAULT_MAX_THREADS 3

//------------------------------------------------------------------------
// Test TBB mutexes when used with parallel_for.h
//
// Usage: test_Mutex.exe [-v] nthread
//
// The -v option causes timing information to be printed.
//
// Compile with _OPENMP and -openmp
//------------------------------------------------------------------------
#include "../test/harness_defs.h"
#include "tbb/atomic.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"
#include "../test/harness.h"
#include "spin_rw_mutex_v2.h"
#include <cstdlib>
#include <cstdio>

// This test deliberately avoids a "using tbb" statement,
// so that the error of putting types in the wrong namespace will be caught.

template<typename M>
struct Counter {
    typedef M mutex_type;
    M mutex;
    volatile long value;
};

//! Function object for use with parallel_for.h.
template<typename C>
struct AddOne: NoAssign {
    C& counter;
    /** Increments counter once for each iteration in the iteration space. */
    void operator()( tbb::blocked_range<size_t>& range ) const {
        for( size_t i=range.begin(); i!=range.end(); ++i ) {
            if( i&1 ) {
                // Try implicit acquire and explicit release
                typename C::mutex_type::scoped_lock lock(counter.mutex);
                counter.value = counter.value+1;
                lock.release();
            } else {
                // Try explicit acquire and implicit release
                typename C::mutex_type::scoped_lock lock;
                lock.acquire(counter.mutex);
                counter.value = counter.value+1;
            }
        }
    }
    AddOne( C& counter_ ) : counter(counter_) {}
};

//! Generic test of a TBB mutex type M.
/** Does not test features specific to reader-writer locks. */
template<typename M>
void Test( const char * name ) {
    if( Verbose ) {
        printf("%s time = ",name);
        fflush(stdout);
    }
    Counter<M> counter;
    counter.value = 0;
    const int n = 100000;
    tbb::tick_count t0 = tbb::tick_count::now();
    tbb::parallel_for(tbb::blocked_range<size_t>(0,n,n/10),AddOne<Counter<M> >(counter));
    tbb::tick_count t1 = tbb::tick_count::now();
    if( Verbose )
        printf("%g usec\n",(t1-t0).seconds());
    if( counter.value!=n )
        printf("ERROR for %s: counter.value=%ld\n",name,counter.value);
}

template<typename M, size_t N>
struct Invariant {
    typedef M mutex_type;
    M mutex;
    const char* mutex_name;
    volatile long value[N];
    Invariant( const char* mutex_name_ ) :
        mutex_name(mutex_name_)
    {
        for( size_t k=0; k<N; ++k )
            value[k] = 0;
    }
    void update() {
        for( size_t k=0; k<N; ++k )
            ++value[k];
    }
    bool value_is( long expected_value ) const {
        long tmp;
        for( size_t k=0; k<N; ++k )
            if( (tmp=value[k])!=expected_value ) {
                printf("ERROR: %ld!=%ld\n", tmp, expected_value);
                return false;
            }
        return true;
    }
    bool is_okay() {
        return value_is( value[0] );
    }
};

//! Function object for use with parallel_for.h.
template<typename I>
struct TwiddleInvariant: NoAssign {
    I& invariant;
    TwiddleInvariant( I& invariant_ ) : invariant(invariant_) {}

    /** Increments counter once for each iteration in the iteration space. */
    void operator()( tbb::blocked_range<size_t>& range ) const {
        for( size_t i=range.begin(); i!=range.end(); ++i ) {
            //! Every 8th access is a write access
            const bool write = (i%8)==7;
            bool okay = true;
            bool lock_kept = true;
            if( (i/8)&1 ) {
                // Try implicit acquire and explicit release
                typename I::mutex_type::scoped_lock lock(invariant.mutex,write);
                execute_aux(lock, i, write, okay, lock_kept);
                lock.release();
            } else {
                // Try explicit acquire and implicit release
                typename I::mutex_type::scoped_lock lock;
                lock.acquire(invariant.mutex,write);
                execute_aux(lock, i, write, okay, lock_kept);
            }
            if( !okay ) {
                printf( "ERROR for %s at %ld: %s %s %s %s\n",invariant.mutex_name, long(i),
                        write     ? "write,"                  : "read,",
                        write     ? (i%16==7?"downgrade,":"") : (i%8==3?"upgrade,":""),
                        lock_kept ? "lock kept,"              : "lock not kept,", // TODO: only if downgrade/upgrade
                        (i/8)&1   ? "impl/expl"               : "expl/impl" );
            }
        }
    }
private:
    void execute_aux(typename I::mutex_type::scoped_lock & lock, const size_t i, const bool write, bool & okay, bool & lock_kept) const {
        if( write ) {
            long my_value = invariant.value[0];
            invariant.update();
            if( i%16==7 ) {
                lock_kept = lock.downgrade_to_reader();
                if( !lock_kept )
                    my_value = invariant.value[0] - 1;
                okay = invariant.value_is(my_value+1);
            }
        } else {
            okay = invariant.is_okay();
            if( i%8==3 ) {
                long my_value = invariant.value[0];
                lock_kept = lock.upgrade_to_writer();
                if( !lock_kept )
                    my_value = invariant.value[0];
                invariant.update();
                okay = invariant.value_is(my_value+1);
            }
        }
    }
};

/** This test is generic so that we can test any other kinds of ReaderWriter locks we write later. */
template<typename M>
void TestReaderWriterLock( const char * mutex_name ) {
    if( Verbose ) {
        printf("%s readers & writers time = ",mutex_name);
        fflush(stdout);
    }
    Invariant<M,8> invariant(mutex_name);
    const size_t n = 500000;
    tbb::tick_count t0 = tbb::tick_count::now();
    tbb::parallel_for(tbb::blocked_range<size_t>(0,n,n/100),TwiddleInvariant<Invariant<M,8> >(invariant));
    tbb::tick_count t1 = tbb::tick_count::now();
    // There is either a writer or a reader upgraded to a writer for each 4th iteration
    long expected_value = n/4;
    if( !invariant.value_is(expected_value) )
        printf("ERROR for %s: final invariant value is wrong\n",mutex_name);
    if( Verbose )
        printf("%g usec\n", (t1-t0).seconds());
}

/** Test try_acquire functionality of a non-reenterable mutex */
template<typename M>
void TestTryAcquire_OneThread( const char * mutex_name ) {
    M tested_mutex;
    typename M::scoped_lock lock1;
    if( lock1.try_acquire(tested_mutex) )
        lock1.release();
    else
        printf("ERROR for %s: try_acquire failed though it should not\n", mutex_name);
    {
        typename M::scoped_lock lock2(tested_mutex);
        if( lock1.try_acquire(tested_mutex) )
            printf("ERROR for %s: try_acquire succeeded though it should not\n", mutex_name);
    }
    if( lock1.try_acquire(tested_mutex) )
        lock1.release();
    else
        printf("ERROR for %s: try_acquire failed though it should not\n", mutex_name);
}

#include "tbb/task_scheduler_init.h"

int TestMain () {
    for( int p=MinThread; p<=MaxThread; ++p ) {
        tbb::task_scheduler_init init( p );
        if( Verbose )
            printf( "testing with %d workers\n", static_cast<int>(p) );
        const int n = 3;
        // Run each test several times.
        for( int i=0; i<n; ++i ) {
            Test<tbb::spin_rw_mutex>( "Spin RW Mutex" );
            TestTryAcquire_OneThread<tbb::spin_rw_mutex>("Spin RW Mutex"); // only tests try_acquire for writers
            TestReaderWriterLock<tbb::spin_rw_mutex>( "Spin RW Mutex" );
        }
        if( Verbose )
            printf( "calling destructor for task_scheduler_init\n" );
    }
    return Harness::Done;
}
