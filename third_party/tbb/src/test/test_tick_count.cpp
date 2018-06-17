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

#include "tbb/tick_count.h"
#include "harness_assert.h"

//! Assert that two times in seconds are very close.
void AssertNear( double x, double y ) {
    ASSERT( -1.0E-10 <= x-y && x-y <=1.0E-10, NULL );
}

//! Test arithmetic operators on tick_count::interval_t
void TestArithmetic( const tbb::tick_count& t0, const tbb::tick_count& t1, const tbb::tick_count& t2 ) {
    tbb::tick_count::interval_t i= t1-t0;
    tbb::tick_count::interval_t j = t2-t1;
    tbb::tick_count::interval_t k = t2-t0;
    AssertSameType( tbb::tick_count::interval_t(), i-j );
    AssertSameType( tbb::tick_count::interval_t(), i+j );
    ASSERT( i.seconds()>1E-9, NULL );
    ASSERT( j.seconds()>1E-9, NULL );
    ASSERT( k.seconds()>2E-9, NULL );
    AssertNear( (i+j).seconds(), k.seconds() );
    AssertNear( (k-j).seconds(), i.seconds() );
    AssertNear( ((k-j)+(j-i)).seconds(), k.seconds()-i.seconds() );
    tbb::tick_count::interval_t sum;
    sum += i;
    sum += j;
    AssertNear( sum.seconds(), k.seconds() );
    sum -= i;
    AssertNear( sum.seconds(), j.seconds() );
    sum -= j;
    AssertNear( sum.seconds(), 0.0 );
}

//------------------------------------------------------------------------
// Test for overhead in calls to tick_count
//------------------------------------------------------------------------

//! Wait for given duration.
/** The duration parameter is in units of seconds. */
static void WaitForDuration( double duration ) {
    tbb::tick_count start = tbb::tick_count::now();
    while( (tbb::tick_count::now()-start).seconds() < duration )
        continue;
}

#include "harness.h"

//! Test that average timer overhead is within acceptable limit.
/** The 'tolerance' value inside the test specifies the limit. */
void TestSimpleDelay( int ntrial, double duration, double tolerance ) {
    double total_worktime = 0;
    // Iteration -1 warms up the code cache.
    for( int trial=-1; trial<ntrial; ++trial ) {
        tbb::tick_count t0 = tbb::tick_count::now();
        if( duration ) WaitForDuration(duration);
        tbb::tick_count t1 = tbb::tick_count::now();
        if( trial>=0 ) {
            total_worktime += (t1-t0).seconds();
        }
    }
    // Compute average worktime and average delta
    double worktime = total_worktime/ntrial;
    double delta = worktime-duration;
    REMARK("worktime=%g delta=%g tolerance=%g\n", worktime, delta, tolerance);

    // Check that delta is acceptable
    if( delta<0 )
        REPORT("ERROR: delta=%g < 0\n",delta);
    if( delta>tolerance )
        REPORT("%s: delta=%g > %g=tolerance where duration=%g\n",delta>3*tolerance?"ERROR":"Warning",delta,tolerance,duration);
}

//------------------------------------------------------------------------
// Test for subtracting calls to tick_count from different threads.
//------------------------------------------------------------------------

#include "tbb/atomic.h"
static tbb::atomic<int> Counter1, Counter2;
static tbb::atomic<bool> Flag1, Flag2;
static tbb::tick_count *tick_count_array;
static double barrier_time;

struct TickCountDifferenceBody {
    TickCountDifferenceBody( int num_threads ) {
        Counter1 = Counter2 = num_threads;
        Flag1 = Flag2 = false;
    }
    void operator()( int id ) const {
        bool last = false;
        // The first barrier.
        if ( --Counter1 == 0 ) last = true;
        while ( !last && !Flag1.load<tbb::acquire>() ) __TBB_Pause( 1 );
        // Save a time stamp of the first barrier releasing.
        tick_count_array[id] = tbb::tick_count::now();

        // The second barrier.
        if ( --Counter2 == 0 ) Flag2.store<tbb::release>(true);
        // The last thread should release threads from the first barrier after it reaches the second
        // barrier to avoid a deadlock.
        if ( last ) Flag1.store<tbb::release>(true);
        // After the last thread releases threads from the first barrier it waits for a signal from
        // the second barrier.
        while ( !Flag2.load<tbb::acquire>() ) __TBB_Pause( 1 );

        if ( last )
            // We suppose that the barrier time is a time interval between the moment when the last
            // thread reaches the first barrier and the moment when the same thread is released from
            // the second barrier. This time is not accurate time of two barriers but it is
            // guaranteed that it does not exceed it.
            barrier_time = (tbb::tick_count::now() - tick_count_array[id]).seconds() / 2;
    }
    ~TickCountDifferenceBody() {
        ASSERT( Counter1 == 0 && Counter2 == 0, NULL );
    }
};

//! Test that two tick_count values recorded on different threads can be meaningfully subtracted.
void TestTickCountDifference( int n ) {
    const double tolerance = 3E-4;
    tick_count_array = new tbb::tick_count[n];

    int num_trials = 0;
    tbb::tick_count start_time = tbb::tick_count::now();
    do {
        NativeParallelFor( n, TickCountDifferenceBody( n ) );
        if ( barrier_time > tolerance )
            // The machine seems to be oversubscibed so skip the test.
            continue;
        for ( int i = 0; i < n; ++i ) {
            for ( int j = 0; j < i; ++j ) {
                double diff = (tick_count_array[i] - tick_count_array[j]).seconds();
                if ( diff < 0 ) diff = -diff;
                if ( diff > tolerance )
                    REPORT( "Warning: cross-thread tick_count difference = %g > %g = tolerance\n", diff, tolerance );
                ASSERT( diff < 3 * tolerance, "Too big difference." );
            }
        }
        // During 5 seconds we are trying to get 10 successful trials.
    } while ( ++num_trials < 10 && (tbb::tick_count::now() - start_time).seconds() < 5 );
    REMARK( "Difference test time: %g sec\n", (tbb::tick_count::now() - start_time).seconds() );
    ASSERT( num_trials == 10, "The machine seems to be heavily oversubscibed, difference test was skipped." );
    delete[] tick_count_array;
}

void TestResolution() {
    static double target_value = 0.314159265358979323846264338327950288419;
    static double step_value = 0.00027182818284590452353602874713526624977572;
    static int range_value = 100;
    double avg_diff = 0.0;
    double max_diff = 0.0;
    for( int i = -range_value; i <= range_value; ++i ) {
        double my_time = target_value + step_value * i;
        tbb::tick_count::interval_t t0(my_time);
        double interval_time = t0.seconds();
        avg_diff += (my_time - interval_time);
        if ( max_diff < my_time-interval_time) max_diff = my_time-interval_time;
        // time always truncates
        ASSERT(interval_time >= 0 && my_time - interval_time < tbb::tick_count::resolution(), "tick_count resolution out of range");
    }
    avg_diff = (avg_diff/(2*range_value+1))/tbb::tick_count::resolution();
    max_diff /= tbb::tick_count::resolution();
    REMARK("avg_diff = %g ticks, max_diff = %g ticks\n", avg_diff, max_diff);
}

#include "tbb/tbb_thread.h"

int TestMain () {
    // Increased tolerance for Virtual Machines
    double tolerance_multiplier = Harness::GetEnv( "VIRTUAL_MACHINE" ) ? 50. : 1.;
    REMARK( "tolerance_multiplier = %g \n", tolerance_multiplier );

    tbb::tick_count t0 = tbb::tick_count::now();
    TestSimpleDelay(/*ntrial=*/1000000,/*duration=*/0,    /*tolerance=*/2E-6 * tolerance_multiplier);
    tbb::tick_count t1 = tbb::tick_count::now();
    TestSimpleDelay(/*ntrial=*/1000,   /*duration=*/0.001,/*tolerance=*/5E-6 * tolerance_multiplier);
    tbb::tick_count t2 = tbb::tick_count::now();
    TestArithmetic(t0,t1,t2);

    TestResolution();

    int num_threads = tbb::tbb_thread::hardware_concurrency();
    ASSERT( num_threads > 0, "tbb::thread::hardware_concurrency() has returned an incorrect value" );
    if ( num_threads > 1 ) {
        REMARK( "num_threads = %d\n", num_threads );
        TestTickCountDifference( num_threads );
    } else {
        REPORT( "Warning: concurrency is too low for TestTickCountDifference ( num_threads = %d )\n", num_threads );
    }

    return Harness::Done;
}
