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

/**
    The test checks that for different ranges of random numbers (from 0 to
    [MinThread, MaxThread]) generated with different seeds the probability
    of each number in the range deviates from the ideal random distribution
    by no more than AcceptableDeviation percent.
**/

#define HARNESS_DEFAULT_MIN_THREADS 2
#define HARNESS_DEFAULT_MAX_THREADS 32

#define HARNESS_DEFINE_PRIVATE_PUBLIC 1
#include "harness_inject_scheduler.h"

#define TEST_TOTAL_SEQUENCE 0

#include "harness.h"
#include "tbb/atomic.h"

//! Coefficient defining tolerable deviation from ideal random distribution
const double AcceptableDeviation = 2.1;
//! Tolerable probability of failure to achieve tolerable distribution
const double AcceptableProbabilityOfOutliers = 1e-5;
//! Coefficient defining the length of random numbers series used to estimate the distribution
/** Number of random values generated per each range element. I.e. the larger is
    the range, the longer is the series of random values. **/
const uintptr_t SeriesBaseLen = 100;
//! Number of random numbers series to generate
const uintptr_t NumSeries = 100;
//! Number of random number generation series with different seeds
const uintptr_t NumSeeds = 100;

tbb::atomic<uintptr_t> NumHighOutliers;
tbb::atomic<uintptr_t> NumLowOutliers;

inline void CheckProbability ( double probability, double expectedProbability, int index, int numIndices, void* seed ) {
    double lowerBound = expectedProbability / AcceptableDeviation,
           upperBound = expectedProbability * AcceptableDeviation;
    if ( probability < lowerBound ) {
        if ( !NumLowOutliers )
            REMARK( "Warning: Probability %.3f of hitting index %d among %d elements is out of acceptable range (%.3f - %.3f) for seed %p\n",
                    probability, index, numIndices, lowerBound, upperBound, seed );
        ++NumLowOutliers;
    }
    else if ( probability > upperBound ) {
        if ( !NumHighOutliers )
            REMARK( "Warning: Probability %.3f of hitting index %d among %d elements is out of acceptable range (%.3f - %.3f) for seed %p\n",
                    probability, index, numIndices, lowerBound, upperBound, seed );
        ++NumHighOutliers;
    }
}

struct CheckDistributionBody {
    void operator() ( int id ) const {
        uintptr_t randomRange = id + MinThread;
        uintptr_t *curHits = new uintptr_t[randomRange]
#if TEST_TOTAL_SEQUENCE
                , *totalHits = new uintptr_t[randomRange]
#endif
        ;
        double expectedProbability = 1./randomRange;
        // Loop through different seeds
        for ( uintptr_t i = 0; i < NumSeeds; ++i ) {
            // Seed value mimics the one used by the TBB task scheduler
            void* seed = (char*)&curHits + i * 16;
            tbb::internal::FastRandom random( seed );
            // According to Section 3.2.1.2 of Volume 2 of Knuth's Art of Computer Programming
            // the following conditions must be hold for m=2^32:
            ASSERT((random.c&1)!=0, "c is relatively prime to m");
            ASSERT((random.a-1)%4==0, "a-1 is a multiple of p, for every prime p dividing m."
                   " And a-1 is a multiple of 4, if m is a multiple of 4");

            memset( curHits, 0, randomRange * sizeof(uintptr_t) );
#if TEST_TOTAL_SEQUENCE
            memset( totalHits, 0, randomRange * sizeof(uintptr_t) );
#endif
            const uintptr_t seriesLen = randomRange * SeriesBaseLen,
                            experimentLen = NumSeries * seriesLen;
            uintptr_t *curSeries = new uintptr_t[seriesLen],  // circular buffer
                       randsGenerated = 0;
            // Initialize statistics
            while ( randsGenerated < seriesLen ) {
                uintptr_t idx = random.get() % randomRange;
                ++curHits[idx];
#if TEST_TOTAL_SEQUENCE
                ++totalHits[idx];
#endif
                curSeries[randsGenerated++] = idx;
            }
            while ( randsGenerated < experimentLen ) {
                for ( uintptr_t j = 0; j < randomRange; ++j ) {
                    CheckProbability( double(curHits[j])/seriesLen, expectedProbability, j, randomRange, seed );
#if TEST_TOTAL_SEQUENCE
                    CheckProbability( double(totalHits[j])/randsGenerated, expectedProbability, j, randomRange, seed );
#endif
                }
                --curHits[curSeries[randsGenerated % seriesLen]];
                int idx = random.get() % randomRange;
                ++curHits[idx];
#if TEST_TOTAL_SEQUENCE
                ++totalHits[idx];
#endif
                curSeries[randsGenerated++ % seriesLen] = idx;
            }
            delete [] curSeries;
        }
        delete [] curHits;
#if TEST_TOTAL_SEQUENCE
        delete [] totalHits;
#endif
    }
};

struct rng {
    tbb::internal::FastRandom my_fast_random;
    rng (unsigned seed):my_fast_random(seed) {}
    unsigned short operator()(){return my_fast_random.get();}
};

template <std::size_t seriesLen >
struct SingleCheck{
    bool operator()(unsigned seed)const{
        std::size_t series1[seriesLen]={0};
        std::size_t series2[seriesLen]={0};
        std::generate(series1,series1+seriesLen,rng(seed));
        std::generate(series2,series2+seriesLen,rng(seed));
        return std::equal(series1,series1+seriesLen,series2);
    }
};

template <std::size_t seriesLen ,size_t seedsNum>
struct CheckReproducibilityBody:NoAssign{
    unsigned short seeds[seedsNum];
    const std::size_t grainSize;
    CheckReproducibilityBody(std::size_t GrainSize): grainSize(GrainSize){
       //first generate seeds to check on, and make sure that sequence is reproducible
       ASSERT(SingleCheck<seedsNum>()(0),"Series generated by FastRandom must be reproducible");
       std::generate(seeds,seeds+seedsNum,rng(0));
    }

    void operator()(int id)const{
       for (size_t i=id*grainSize; (i<seedsNum)&&(i< ((id+1)*grainSize));++i ){
           ASSERT(SingleCheck<seriesLen>()(i),"Series generated by FastRandom must be reproducible");
       }
    }

};
#include "tbb/tbb_thread.h"

int TestMain () {
    ASSERT( AcceptableDeviation < 100, NULL );
    MinThread = max(MinThread, 2);
    MaxThread = max(MinThread, MaxThread);
    double NumChecks = double(NumSeeds) * (MaxThread - MinThread + 1) * (MaxThread + MinThread) / 2.0 * (SeriesBaseLen * NumSeries - SeriesBaseLen);
    REMARK( "Number of distribution quality checks %g\n", NumChecks );
    NumLowOutliers = NumHighOutliers = 0;
    // Parallelism is used in this test only to speed up the long serial checks
    // Essentially it is a loop over random number ranges
    // Ideally tbb::parallel_for could be used to parallelize the outermost loop
    // in CheckDistributionBody, but it is not used to avoid unit test contamination.
    int P = tbb::tbb_thread::hardware_concurrency();
    enum {reproducibilitySeedsToTest=1000};
    enum {reproducibilitySeriesLen=100};
    CheckReproducibilityBody<reproducibilitySeriesLen,reproducibilitySeedsToTest>  CheckReproducibility(reproducibilitySeedsToTest/MaxThread);
    while ( MinThread <= MaxThread ) {
        int ThreadsToRun = min(P, MaxThread - MinThread + 1);
        REMARK("Checking random range [%d;%d)\n", MinThread, MinThread+ThreadsToRun);
        NativeParallelFor( ThreadsToRun, CheckDistributionBody() );
        NativeParallelFor( ThreadsToRun, CheckReproducibility );
        MinThread += P;
    }
    double observedProbabilityOfOutliers = (NumLowOutliers + NumHighOutliers) / NumChecks;
    if ( observedProbabilityOfOutliers > AcceptableProbabilityOfOutliers ) {
        if ( NumLowOutliers )
            REPORT( "Warning: %d cases of too low probability of a given number detected\n", (int)NumLowOutliers );
        if ( NumHighOutliers )
            REPORT( "Warning: %d cases of too high probability of a given number detected\n", (int)NumHighOutliers );
        ASSERT( observedProbabilityOfOutliers <= AcceptableProbabilityOfOutliers, NULL );
    }
    return Harness::Done;
}
