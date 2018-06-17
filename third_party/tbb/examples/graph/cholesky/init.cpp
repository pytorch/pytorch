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

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <mkl_cblas.h>

static void posdef_gen( double * A, int n )
{
    /* Allocate memory for the matrix and its transpose */
    double *L = (double *)calloc( sizeof( double ), n*n );
    assert( L );

    double *LT = (double *)calloc( sizeof( double ), n*n) ;
    assert( LT );

    memset( A, 0, sizeof( double )*n*n );

    /* Generate a conditioned matrix and fill it with random numbers */
    for ( int j = 0; j < n; ++j ) {
        for ( int k = 0; k < j; ++k ) {
            // The initial value has to be between [0,1].
            L[k*n+j] = ( ( (j*k) / ((double)(j+1)) / ((double)(k+2)) * 2.0) - 1.0 ) / ((double)n);
        }

        L[j*n+j] = 1;
    }

    /* Compute transpose of the matrix */
    for ( int i = 0; i < n; ++i ) {
        for ( int j = 0; j < n; ++j ) {
            LT[j*n+i] = L[i*n+j];
        }
    }
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, L, n, LT, n, 0, A, n );

    free( L );
    free( LT );
}

// Read the matrix from the input file
void matrix_init( double * &A, int &n, const char *fname ) {
    if( fname ) {
        int i;
        int j;
        FILE *fp;

        fp = fopen( fname, "r" );
        if ( fp == NULL ) {
            fprintf( stderr, "\nFile does not exist\n" );
            exit( 0 );
        }
        if ( fscanf( fp, "%d", &n ) <= 0 ) {
            fprintf( stderr,"\nCouldn't read n from %s\n", fname );
            exit( 1 );
        }
        A = (double *)calloc( sizeof( double ), n*n );
        for ( i = 0; i < n; ++i ) {
            for ( j = 0; j <= i; ++j ) {
                if( fscanf( fp, "%lf ", &A[i*n+j] ) <= 0) {
                    fprintf( stderr,"\nMatrix size incorrect %i %i\n", i, j );
                    exit( 1 );
                }
                if ( i != j ) {
                    A[j*n+i] = A[i*n+j];
                }
            }
        }
        fclose( fp );
    } else {
        A = (double *)calloc( sizeof( double ), n*n );
        posdef_gen( A, n );
    }
}

// write matrix to file
void matrix_write ( double *A, int n, const char *fname, bool is_triangular = false )
{
    if( fname ) {
        int i = 0;
        int j = 0;
        FILE *fp = NULL;

        fp = fopen( fname, "w" );
        if ( fp == NULL ) {
            fprintf( stderr, "\nCould not open file %s for writing.\n", fname );
            exit( 0 );
        }
        fprintf( fp, "%d\n", n );
        for ( i = 0; i < n; ++i) {
            for ( j = 0; j <= i; ++j ) {
                fprintf( fp, "%lf ", A[j*n+i] );
            }
            if ( !is_triangular ) {
                for ( ; j < n; ++j ) {
                    fprintf( fp, "%lf ", A[i*n+j] );
                }
            } else {
                for ( ; j < n; ++j ) {
                    fprintf( fp, "%lf ", 0.0 );
                }
            } 
            fprintf( fp, "\n" );
        }
        if ( is_triangular ) {
            fprintf( fp, "\n" );
            for ( i = 0; i < n; ++i ) {
                for ( j = 0; j < i; ++j ) {
                    fprintf( fp, "%lf ", 0.0 );
                }
                for ( ; j < n; ++j ) {
                    fprintf( fp, "%lf ", A[i*n+j] );
                }
                fprintf( fp, "\n" );
            }
        }
        fclose( fp );
    }
}
