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

// Example program that computes number of prime numbers up to n, 
// where n is a command line argument.  The algorithm here is a 
// fairly efficient version of the sieve of Eratosthenes. 
// The parallel version demonstrates how to use parallel_reduce,
// and in particular how to exploit lazy splitting.

#include "primes.h"

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (target(mic))
#endif // __TBB_MIC_OFFLOAD
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <math.h>
#include <cstdlib>
#include <cctype>
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"

using namespace std;

//! If true, then print primes on stdout.
static bool printPrimes = false;


class Multiples {
    inline NumberType strike( NumberType start, NumberType limit, NumberType stride ) {
        // Hoist "my_is_composite" into register for sake of speed.
        bool* is_composite = my_is_composite;
        assert( stride>=2 );
        for( ;start<limit; start+=stride ) 
            is_composite[start] = true;
        return start;
    }
    //! Window into conceptual sieve 
    bool* my_is_composite;

    //! Indexes into window
    /** my_striker[k] is an index into my_composite corresponding to
        an odd multiple multiple of my_factor[k]. */
    NumberType* my_striker;

    //! Prime numbers less than m.
    NumberType* my_factor;
public:
    //! NumberType of factors in my_factor.
    NumberType n_factor;
    NumberType m;
    Multiples( NumberType n ) {
        m = NumberType(sqrt(double(n)));
        // Round up to even
        m += m&1;
        my_is_composite = new bool[m/2];
        my_striker = new NumberType[m/2];
        my_factor = new NumberType[m/2];
        n_factor = 0;
        memset( my_is_composite, 0, m/2 );
        for( NumberType i=3; i<m; i+=2 ) {
            if( !my_is_composite[i/2] ) {
                if( printPrimes )
                    printf("%d\n",(int)i);
                my_striker[n_factor] = strike( i/2, m/2, i );
                my_factor[n_factor++] = i;
            }
        }
    }

    //! Find primes in range [start,window_size), advancing my_striker as we go.
    /** Returns number of primes found. */
    NumberType find_primes_in_window( NumberType start, NumberType window_size ) {
        bool* is_composite = my_is_composite;
        memset( is_composite, 0, window_size/2 );
        for( size_t k=0; k<n_factor; ++k )
            my_striker[k] = strike( my_striker[k]-m/2, window_size/2, my_factor[k] );
        NumberType count = 0;
        for( NumberType k=0; k<window_size/2; ++k ) {
            if( !is_composite[k] ) {
                if( printPrimes )
                    printf("%ld\n",long(start+2*k+1));
                ++count;
            }
        }
        return count;
    }

    ~Multiples() {
        delete[] my_factor;
        delete[] my_striker;
        delete[] my_is_composite;
    }

    //------------------------------------------------------------------------
    // Begin extra members required by parallel version
    //------------------------------------------------------------------------

    // Splitting constructor
    Multiples( const Multiples& f, tbb::split ) :
        n_factor(f.n_factor),
        m(f.m),
        my_is_composite(NULL),
        my_striker(NULL),
        my_factor(f.my_factor)
    {}

    bool is_initialized() const {
        return my_is_composite!=NULL;
    }

    void initialize( NumberType start ) { 
        assert( start>=1 );
        my_is_composite = new bool[m/2];
        my_striker = new NumberType[m/2];
        for( size_t k=0; k<n_factor; ++k ) {
            NumberType f = my_factor[k];
            NumberType p = (start-1)/f*f % m;
            my_striker[k] = (p&1 ? p+2*f : p+f)/2;
            assert( m/2<=my_striker[k] );
        }
    }

    // Move other to *this.
    void move( Multiples& other ) {
        // The swap moves the contents of other to *this and causes the old contents
        // of *this to be deleted later when other is destroyed.
        std::swap( my_striker, other.my_striker );
        std::swap( my_is_composite, other.my_is_composite );
        // other.my_factor is a shared pointer that was copied by the splitting constructor.
        // Set it to NULL to prevent premature deletion by the destructor of ~other.
        assert(my_factor==other.my_factor);
        other.my_factor = NULL;
    }

    //------------------------------------------------------------------------
    // End extra methods required by parallel version
    //------------------------------------------------------------------------
};

//! Count number of primes between 0 and n
/** This is the serial version. */
NumberType SerialCountPrimes( NumberType n ) {
    // Two is special case
    NumberType count = n>=2;
    if( n>=3 ) {
        Multiples multiples(n);
        count += multiples.n_factor;
        if( printPrimes ) 
            printf("---\n");
        NumberType window_size = multiples.m;
        for( NumberType j=multiples.m; j<=n; j+=window_size ) { 
            if( j+window_size>n+1 ) 
                window_size = n+1-j;
            count += multiples.find_primes_in_window( j, window_size );
        }
    }
    return count;
}

//! Range of a sieve window.
class SieveRange {
    //! Width of full-size window into sieve.
    const NumberType my_stride;

    //! Always multiple of my_stride
    NumberType my_begin;

    //! One past last number in window.
    NumberType my_end;

    //! Width above which it is worth forking.
    const NumberType my_grainsize;

    bool assert_okay() const {
        assert( my_begin%my_stride==0 );
        assert( my_begin<=my_end );
        assert( my_stride<=my_grainsize );
        return true;
    } 
public:
    //------------------------------------------------------------------------
    // Begin signatures required by parallel_reduce
    //------------------------------------------------------------------------
    bool is_divisible() const {return my_end-my_begin>my_grainsize;}
    bool empty() const {return my_end<=my_begin;}
    SieveRange( SieveRange& r, tbb::split ) :
        my_stride(r.my_stride), 
        my_grainsize(r.my_grainsize),
        my_end(r.my_end)
    {
        assert( r.is_divisible() );
        assert( r.assert_okay() );
        NumberType middle = r.my_begin + (r.my_end-r.my_begin+r.my_stride-1)/2;
        middle = middle/my_stride*my_stride;
        my_begin = middle;
        r.my_end = middle;
        assert( assert_okay() );
        assert( r.assert_okay() );
    }
    //------------------------------------------------------------------------
    // End of signatures required by parallel_reduce
    //------------------------------------------------------------------------
    NumberType begin() const {return my_begin;}
    NumberType end() const {return my_end;}
    SieveRange( NumberType begin, NumberType end, NumberType stride, NumberType grainsize ) :
        my_begin(begin),
        my_end(end),
        my_stride(stride),      
        my_grainsize(grainsize<stride?stride:grainsize)
    {
        assert( assert_okay() );
    }
};

//! Loop body for parallel_reduce.
/** parallel_reduce splits the sieve into subsieves.
    Each subsieve handles a subrange of [0..n]. */
class Sieve {
public:
    //! Prime Multiples to consider, and working storage for this subsieve.
    ::Multiples multiples;

    //! NumberType of primes found so far by this subsieve.
    NumberType count;

    //! Construct Sieve for counting primes in [0..n].
    Sieve( NumberType n ) :
        multiples(n),
        count(0)
    {}

    //------------------------------------------------------------------------
    // Begin signatures required by parallel_reduce
    //------------------------------------------------------------------------
    void operator()( const SieveRange& r ) {
        NumberType m = multiples.m;
        if( multiples.is_initialized() ) { 
            // Simply reuse "Multiples" structure from previous window
            // This works because parallel_reduce always applies
            // *this from left to right.
        } else {
            // Need to initialize "Multiples" because *this is a forked copy
            // that needs to be set up to start at r.begin().
            multiples.initialize( r.begin() );
        }
        NumberType window_size = m;
        for( NumberType j=r.begin(); j<r.end(); j+=window_size ) { 
            assert( j%multiples.m==0 );
            if( j+window_size>r.end() ) 
                window_size = r.end()-j;
            count += multiples.find_primes_in_window( j, window_size );
        }
    }
    void join( Sieve& other ) {
        count += other.count;
        // Final value of multiples needs to final value of other multiples,
        // so that *this can correctly process next window to right.
        multiples.move( other.multiples );
    }
    Sieve( Sieve& other, tbb::split ) :
        multiples(other.multiples,tbb::split()),
        count(0)
    {}
    //------------------------------------------------------------------------
    // End of signatures required by parallel_reduce
    //------------------------------------------------------------------------
};

//! Count number of primes between 0 and n
/** This is the parallel version. */
NumberType ParallelCountPrimes( NumberType n , int number_of_threads, NumberType grain_size ) {
    tbb::task_scheduler_init init(number_of_threads);

    // Two is special case
    NumberType count = n>=2;
    if( n>=3 ) {
        Sieve s(n);
        count += s.multiples.n_factor;
        if( printPrimes )
            printf("---\n");
        using namespace tbb;
        // Explicit grain size and simple_partitioner() used here instead of automatic grainsize 
        // determination becase we want SieveRange to be decomposed down to grainSize or smaller.  
        // Doing so improves odds that the working set fits in cache when evaluating Sieve::operator().
        parallel_reduce( SieveRange( s.multiples.m, n, s.multiples.m, grain_size ), s, simple_partitioner() );
        count += s.count;
    }
    return count;
}
