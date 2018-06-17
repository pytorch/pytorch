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

#include "tbb/parallel_while.h"
#include "harness.h"

const int N = 200;

typedef int Element;

//! Representation of an array index with only those signatures required by parallel_while.
class MinimalArgumentType {
    void operator=( const MinimalArgumentType& );
    long my_value;
    enum {
        DEAD=0xDEAD,
        LIVE=0x2718,
        INITIALIZED=0x3141
    } my_state;
public:
    ~MinimalArgumentType() {
        ASSERT( my_state==LIVE||my_state==INITIALIZED, NULL );
        my_state = DEAD;
    }
    MinimalArgumentType() {
        my_state = LIVE;
    }
    void set_value( long i ) {
        ASSERT( my_state==LIVE||my_state==INITIALIZED, NULL );
        my_value = i;
        my_state = INITIALIZED;
    }
    long get_value() const {
        ASSERT( my_state==INITIALIZED, NULL );
        return my_value;
    }
};

class IntegerStream {
    long my_limit;
    long my_index;
public:
    IntegerStream( long n ) : my_limit(n), my_index(0) {}
    bool pop_if_present( MinimalArgumentType& v ) {
        if( my_index>=my_limit )
            return false;
        v.set_value( my_index );
        my_index+=2;
        return true;
    }
};

class MatrixMultiplyBody: NoAssign {
    Element (*a)[N];
    Element (*b)[N];
    Element (*c)[N];
    const int n;
    tbb::parallel_while<MatrixMultiplyBody>& my_while;
public:
    typedef MinimalArgumentType argument_type;
    void operator()( argument_type i_arg ) const {
        long i = i_arg.get_value();
        if( (i&1)==0 && i+1<N ) {
            MinimalArgumentType value;
            value.set_value(i+1);
            my_while.add( value );
        }
        for( int j=0; j<n; ++j )
            c[i][j] = 0;
        for( int k=0; k<n; ++k ) {
            Element aik = a[i][k];
            for( int j=0; j<n; ++j )
                c[i][j] += aik*b[k][j];
        }
    }
    MatrixMultiplyBody( tbb::parallel_while<MatrixMultiplyBody>& w, Element c_[N][N], Element a_[N][N], Element b_[N][N], int n_ ) :
        a(a_), b(b_), c(c_), n(n_),  my_while(w)
    {}
};

void WhileMatrixMultiply( Element c[N][N], Element a[N][N], Element b[N][N], int n ) {
    IntegerStream stream( N );
    tbb::parallel_while<MatrixMultiplyBody> w;
    MatrixMultiplyBody body(w,c,a,b,n);
    w.run( stream, body );
}

#include "tbb/tick_count.h"
#include <cstdlib>
#include <cstdio>
using namespace std;

static long Iterations = 5;

static void SerialMatrixMultiply( Element c[N][N], Element a[N][N], Element b[N][N], int n ) {
    for( int i=0; i<n; ++i ) {
        for( int j=0; j<n; ++j )
            c[i][j] = 0;
        for( int k=0; k<n; ++k ) {
            Element aik = a[i][k];
            for( int j=0; j<n; ++j )
                c[i][j] += aik*b[k][j];
        }
    }
}

static void InitializeMatrix( Element x[N][N], int n, int salt ) {
    for( int i=0; i<n; ++i )
        for( int j=0; j<n; ++j )
            x[i][j] = (i*n+j)^salt;
}

static Element A[N][N], B[N][N], C[N][N], D[N][N];

static void Run( int nthread, int n ) {
    /* Initialize matrices */
    InitializeMatrix(A,n,5);
    InitializeMatrix(B,n,10);
    InitializeMatrix(C,n,0);
    InitializeMatrix(D,n,15);

    tbb::tick_count t0 = tbb::tick_count::now();
    for( long i=0; i<Iterations; ++i ) {
        WhileMatrixMultiply( C, A, B, n );
    }
    tbb::tick_count t1 = tbb::tick_count::now();
    SerialMatrixMultiply( D, A, B, n );

    // Check result
    for( int i=0; i<n; ++i )
        for( int j=0; j<n; ++j )
            ASSERT( C[i][j]==D[i][j], NULL );
    REMARK("time=%g\tnthread=%d\tn=%d\n",(t1-t0).seconds(),nthread,n);
}

#include "tbb/task_scheduler_init.h"
#include "harness_cpu.h"

int TestMain () {
    if( MinThread<1 ) {
        REPORT("number of threads must be positive\n");
        exit(1);
    }
    for( int p=MinThread; p<=MaxThread; ++p ) {
        tbb::task_scheduler_init init( p );
        for( int n=N/4; n<=N; n+=N/4 )
            Run(p,n);

        // Test that all workers sleep when no work
        TestCPUUserTime(p);
    }
    return Harness::Done;
}

