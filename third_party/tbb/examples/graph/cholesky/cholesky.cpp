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

#include <string>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <map>

#include "mkl_lapack.h"
#include "mkl.h"

#include "tbb/tbb_config.h"
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"

// Application command line arguments parsing
#include "../../common/utility/utility.h"

/************************************************************
 FORWARD DECLARATIONS
************************************************************/

/**********************************************
 Read or generate a positive-definite matrix
 -- reads from file if fname != NULL
     -- sets n to matrix size
     -- allocates and reads values in to A
 -- otherwise generates a matrix
     -- uses n to determine size
     -- allocates and generates values in to A
**********************************************/
void matrix_init( double * &A, int &n, const char *fname );

/**********************************************
 Writes a lower triangular matrix to a file
 -- first line of file is n 
 -- subsequently 1 row per line
**********************************************/
void matrix_write ( double *A, int n, const char *fname, bool is_triangular = false );

/************************************************************
 GLOBAL VARIABLES
************************************************************/
bool g_benchmark_run = false;
int g_num_tbb_threads = tbb::task_scheduler_init::default_num_threads();
int g_n = -1, g_b = -1, g_num_trials = 1;
char *g_input_file_name = NULL;
char *g_output_prefix = NULL;
std::string g_alg_name;

// Creates tiled array
static double ***create_tile_array( double *A, int n, int b ) {
    const int p = n/b;
    double ***tile = (double ***)calloc( sizeof( double ** ), p );

    for ( int j = 0; j < p; ++j ) {
        tile[j] = (double **)calloc( sizeof( double * ), p );
    }

    for ( int j = 0; j < p; ++j ) {
        for ( int i = 0; i < p; ++i ) {
            double *temp_block = (double *)calloc( sizeof( double ), b*b );

            for ( int A_j = j*b, T_j = 0; T_j < b; ++A_j, ++T_j ) {
                for ( int A_i = i*b, T_i = 0; T_i < b; ++A_i, ++T_i ) {
                    temp_block[T_j*b+T_i] = A[A_j*n+A_i];
                }
            }

            tile[j][i] = temp_block;
        }
    }
    return tile;
}

static void collapse_tile_array( double ***tile, double *A, int n, int b ) {
    const int p = n/b;

    for ( int j = 0; j < p; ++j ) {
        for ( int i = 0; i < p; ++i ) {
            double *temp_block = tile[j][i];

            for ( int A_j = j*b, T_j = 0; T_j < b; ++A_j, ++T_j ) {
                for ( int A_i = i*b, T_i = 0; T_i < b; ++A_i, ++T_i ) {
                    A[A_j*n+A_i] = temp_block[T_j*b+T_i];
                }
            }

            free( temp_block );
            tile[j][i] = NULL;
        }

        free( tile[j] );
    }

    free( tile );
}

/************************************************************
 Helper base class: algorithm
************************************************************/
class algorithm {

    std::string name;
    bool is_tiled;

    bool check_if_valid( double *A0, double *C, double *A, int n ) {
        char transa = 'n', transb = 't';
        double alpha = 1;
        double beta = 0;

        for ( int i = 0; i < n; ++i ) {
            for ( int j = i+1; j < n; ++j ) {
                A0[j*n+i] = 0.;
            }
        }

        dgemm ( &transa, &transb, &n, &n, &n, &alpha, A0, &n, A0, &n, &beta, C, &n );

        for ( int j = 0; j < n; ++j ) {
            for ( int i = 0; i < n; ++i ) {
                const double epsilon = std::abs( A[j*n+i]*0.1 );

                if ( std::abs( C[j*n+i] - A[j*n+i] ) > epsilon ) {
                    printf( "ERROR: %s did not validate at C(%d,%d) = %lf != A(%d,%d) = %lf\n",
                        name.c_str(), i, j, C[j*n+i], i, j, A[j*n+i] );
                    printf( "ERROR: %g; %g < %g < %g\n", epsilon, A[j*n+i] - epsilon, C[j*n+i], A[j*n+i] + epsilon );
                    return false;
                }
            }
        }
        return true;
    }

public:
    algorithm( const std::string& alg_name, bool t ) : name(alg_name), is_tiled(t) {}

    double operator() ( double *A, int n, int b, int trials ) {
        tbb::tick_count t0, t1;
        double elapsed_time = 0.0;
        double *A0 = (double *)calloc( sizeof( double ), n*n );
        double *C = (double *)calloc( sizeof( double ), n*n );

        for ( int t = 0; t < trials+1; ++t ) {
            if ( is_tiled ) {
                double ***tile = create_tile_array( A, n, b );
                t0 = tbb::tick_count::now();
                func( tile, n, b );
                t1 = tbb::tick_count::now();

                collapse_tile_array( tile, A0, n, b );
            }
            else {
                memcpy( A0, A, sizeof( double )*n*n );
                t0 = tbb::tick_count::now();
                func( A0, n, b );
                t1 = tbb::tick_count::now();
            }

            if ( t ) elapsed_time += (t1-t0).seconds();

            if( !g_benchmark_run && !check_if_valid( A0, C, A, n ) ) {
                if ( g_output_prefix ) {
                    std::string s( g_output_prefix );
                    s += "_" + name + ".txt";
                    matrix_write( A0, g_n, s.c_str(), true );
                    free( A0 );
                    free( C );
                    return 0.;
                }
            }
        }

        if ( g_output_prefix ) {
            std::string s( g_output_prefix );
            s += "_" + name + ".txt";
            matrix_write( A0, g_n, s.c_str(), true );
        }

        printf( "%s %d %d %d %d %lf %lf\n", name.c_str(), g_num_tbb_threads, trials, n, b, elapsed_time, elapsed_time/trials );
        free( A0 );
        free( C );
        return elapsed_time;
    }

protected:
    // Main algorithm body function must be defined in any direved class
    virtual void func( void * ptr, int n, int b ) = 0;
};

/***********************************************************/

static void call_dpotf2( double ***tile, int b, int k ) {
    double *A_block = tile[k][k];
    char uplo = 'l';
    int info = 0;
    dpotf2( &uplo, &b, A_block, &b, &info ); 
    return;
}

static void call_dtrsm( double ***tile, int b, int k, int j ) {
    double *A_block = tile[k][j];
    double *L_block = tile[k][k];
    char uplo = 'l', side = 'r', transa = 't', diag = 'n';
    double alpha = 1;
    dtrsm( &side, &uplo, &transa, &diag, &b, &b, &alpha, L_block, &b, A_block, &b );
    return;
}

static void call_dsyr2k( double ***tile, int b, int k, int j, int i ) {
    double *A_block = tile[i][j];
    char transa = 'n', transb = 't';
    char uplo = 'l';
    double alpha = -1;
    double beta = 1;

    if ( i == j ) {   // Diagonal block
        double *L_block = tile[k][i];
        dsyrk( &uplo, &transa, &b, &b, &alpha, L_block, &b, &beta, A_block, &b );
    } else {   // Non-diagonal block
        double *L2_block = tile[k][i];
        double *L1_block = tile[k][j];
        dgemm( &transa, &transb, &b, &b, &b, &alpha, L1_block, &b, L2_block, &b, &beta, A_block, &b );
    }
    return;
}

class algorithm_crout : public algorithm
{
public:
    algorithm_crout() : algorithm("crout_cholesky", true) {}

protected:
    virtual void func( void * ptr, int n, int b ) {
        double ***tile = (double ***)ptr;
        const int p = n/b;

        for ( int k = 0; k < p; ++k ) {
            call_dpotf2( tile, b, k );

            for ( int j = k+1; j < p; ++j ) {
                call_dtrsm( tile, b, k, j );

                for ( int i = k+1; i <= j; ++i ) {
                    call_dsyr2k( tile, b, k, j, i );
                }
            }
        }
    }
};

class algorithm_dpotrf : public algorithm
{
public:
    algorithm_dpotrf() : algorithm("dpotrf_cholesky", false) {}

protected:
    virtual void func( void * ptr, int n, int /* b */ ) {
        double *A = (double *)ptr;
        int lda = n;
        int info = 0;
        char uplo = 'l';
        dpotrf( &uplo, &n, A, &lda, &info );
    }
};

/************************************************************
 Begin data join graph based version of cholesky
************************************************************/

typedef union {
    char a[4];
    size_t tag;
} tag_t;

typedef double * tile_t;

typedef std::pair< tag_t, tile_t > tagged_tile_t;
typedef tbb::flow::tuple< tagged_tile_t > t1_t;
typedef tbb::flow::tuple< tagged_tile_t, tagged_tile_t > t2_t;
typedef tbb::flow::tuple< tagged_tile_t, tagged_tile_t, tagged_tile_t > t3_t;

typedef tbb::flow::multifunction_node< tagged_tile_t, t1_t > dpotf2_node_t;
typedef tbb::flow::multifunction_node< t2_t, t2_t > dtrsm_node_t;
typedef tbb::flow::multifunction_node< t3_t, t3_t > dsyr2k_node_t;

typedef tbb::flow::join_node< t2_t, tbb::flow::tag_matching > dtrsm_join_t;
typedef tbb::flow::join_node< t3_t, tbb::flow::tag_matching > dsyr2k_join_t;

class dpotf2_body {
    int p;
    int b;
public:
    dpotf2_body( int p_, int b_ ) : p(p_), b(b_) {}

    void operator()( const tagged_tile_t &in, dpotf2_node_t::output_ports_type &ports ) {
        int k = in.first.a[0];
        tile_t A_block = in.second;
        tag_t t;
        t.tag = 0;
        t.a[0] = k;
        char uplo = 'l';
        int info = 0;
        dpotf2( &uplo, &b, A_block, &b, &info );

        // Send to dtrsms in same column
        // k == k  j == k 
        t.a[2] = k;
        for ( int j = k+1; j < p; ++j ) {
            t.a[1] = j;
            tbb::flow::get<0>( ports ).try_put( std::make_pair( t, A_block ) );
        }
    }
};

class dtrsm_body {
    int p;
    int b;
public:
    dtrsm_body( int p_, int b_ ) : p(p_), b(b_) {}

    void operator()( const t2_t &in, dtrsm_node_t::output_ports_type &ports ) {
        using tbb::flow::get;

        tagged_tile_t in0 = get<0>( in );
        tagged_tile_t in1 = get<1>( in );
        int k = in0.first.a[0];
        int j = in0.first.a[1];
        tile_t L_block = in0.second;
        tile_t A_block = in1.second;
        tag_t t;
        t.tag = 0;
        t.a[0] = k;
        char uplo = 'l', side = 'r', transa = 't', diag = 'n';
        double alpha = 1;
        dtrsm( &side, &uplo, &transa, &diag, &b, &b, &alpha, L_block, &b, A_block, &b);

        // Send to rest of my row
        t.a[1] = j;
        for ( int i = k+1; i <= j; ++i ) {
            t.a[2] = i;
            get<0>( ports ).try_put( std::make_pair( t, A_block ) );
        }

        // Send to transposed row
        t.a[2] = j;
        for ( int i = j; i < p; ++i ) {
            t.a[1] = i;
            get<1>( ports ).try_put( std::make_pair( t, A_block ) );
        }
    }
};

class dsyr2k_body {
    int p;
    int b;
public:
    dsyr2k_body( int p_, int b_ ) : p(p_), b(b_) {}

    void operator()( const t3_t &in, dsyr2k_node_t::output_ports_type &ports ) {
        using tbb::flow::get;

        tag_t t;
        t.tag = 0;
        char transa = 'n', transb = 't';
        char uplo = 'l';
        double alpha = -1;
        double beta = 1;

        tagged_tile_t in0 = get<0>( in );
        tagged_tile_t in1 = get<1>( in );
        tagged_tile_t in2 = get<2>( in );
        int k = in2.first.a[0];
        int j = in2.first.a[1];
        int i = in2.first.a[2];

        tile_t A_block = in2.second; 
        if ( i == j ) {   // Diagonal block
            tile_t L_block = in0.second;
            dsyrk( &uplo, &transa, &b, &b, &alpha, L_block, &b, &beta, A_block, &b );
        } else {   // Non-diagonal block
            tile_t L1_block = in0.second;
            tile_t L2_block = in1.second;
            dgemm( &transa, &transb, &b, &b, &b, &alpha, L1_block, &b, L2_block, &b, &beta, A_block, &b );
        }

        // All outputs flow to next step
        t.a[0] = k+1;
        t.a[1] = j;
        t.a[2] = i;
        if ( k != p-1 && j == k+1 && i == k+1 ) {
            get<0>( ports ).try_put( std::make_pair( t, A_block ) );
        }

        if ( k < p-2 ) {
            if ( i == k+1 && j > i ) {
                t.a[0] = k+1;
                t.a[1] = j;
                get<1>( ports ).try_put( std::make_pair( t, A_block ) );
            }

            if ( j != k+1 && i != k+1 ) {
                t.a[0] = k+1;
                t.a[1] = j;
                t.a[2] = i;
                get<2>( ports ).try_put( std::make_pair( t, A_block ) );
            }
        }
    }
};

struct tagged_tile_to_size_t {
    size_t operator()( const tagged_tile_t &t ) {
        return t.first.tag;
    }
};

class algorithm_join : public algorithm
{
public:
    algorithm_join() : algorithm("data_join_cholesky", true) {}

protected:
    virtual void func( void * ptr, int n, int b ) {
        using tbb::flow::unlimited;
        using tbb::flow::output_port;
        using tbb::flow::input_port;

        double ***tile = (double ***)ptr;
        const int p = n/b;
        tbb::flow::graph g;

        dpotf2_node_t dpotf2_node( g, unlimited, dpotf2_body(p, b) );
        dtrsm_node_t dtrsm_node( g, unlimited, dtrsm_body(p, b) );
        dsyr2k_node_t dsyr2k_node( g, unlimited, dsyr2k_body(p, b) );
        dtrsm_join_t dtrsm_join( g, tagged_tile_to_size_t(), tagged_tile_to_size_t() );
        dsyr2k_join_t dsyr2k_join( g, tagged_tile_to_size_t(), tagged_tile_to_size_t(), tagged_tile_to_size_t() );

        make_edge( output_port<0>( dsyr2k_node ), dpotf2_node );

        make_edge( output_port<0>( dpotf2_node ), input_port<0>( dtrsm_join ) );
        make_edge( output_port<1>( dsyr2k_node ), input_port<1>( dtrsm_join ) );
        make_edge( dtrsm_join, dtrsm_node );

        make_edge( output_port<0>( dtrsm_node ), input_port<0>( dsyr2k_join ) );
        make_edge( output_port<1>( dtrsm_node ), input_port<1>( dsyr2k_join ) );
        make_edge( output_port<2>( dsyr2k_node ), input_port<2>( dsyr2k_join ) );
        make_edge( dsyr2k_join, dsyr2k_node );

        // Now we need to send out the tiles to their first nodes
        tag_t t;
        t.tag = 0;
        t.a[0] = 0;
        t.a[1] = 0;
        t.a[2] = 0;

        // Send to feedback input of first dpotf2
        // k == 0, j == 0, i == 0
        dpotf2_node.try_put( std::make_pair( t, tile[0][0] ) );

        // Send to feedback input (port 1) of each dtrsm
        // k == 0, j == 1..p-1
        for ( int j = 1; j < p; ++j ) {
            t.a[1] = j;
            input_port<1>( dtrsm_join ).try_put( std::make_pair( t, tile[0][j] ) );
        }

        // Send to feedback input (port 2) of each dsyr2k
        // k == 0
        for ( int i = 1; i < p; ++i ) {
            t.a[2] = i;

            for ( int j = i; j < p; ++j ) {
                t.a[1] = j;
                input_port<2>( dsyr2k_join ).try_put( std::make_pair( t, tile[i][j] ) );
            }
        }

        g.wait_for_all();
    }
};

/************************************************************
 End data join graph based version of cholesky
************************************************************/

/************************************************************
 Begin dependence graph based version of cholesky
************************************************************/

typedef tbb::flow::continue_node< tbb::flow::continue_msg > continue_type;
typedef continue_type * continue_ptr_type;

#if !__TBB_CPP11_LAMBDAS_PRESENT
// Using helper functor classes (instead of built-in C++ 11 lambda functions)
class call_dpotf2_functor
{
    double ***tile;
    int b, k;
public:
    call_dpotf2_functor( double ***tile_, int b_, int k_ )
        : tile(tile_), b(b_), k(k_) {}

    void operator()( const tbb::flow::continue_msg & ) { call_dpotf2( tile, b, k ); }
};

class call_dtrsm_functor
{
    double ***tile;
    int b, k, j;
public:
    call_dtrsm_functor( double ***tile_, int b_, int k_, int j_ )
        : tile(tile_), b(b_), k(k_), j(j_) {}

    void operator()( const tbb::flow::continue_msg & ) { call_dtrsm( tile, b, k, j ); }
};

class call_dsyr2k_functor
{
    double ***tile;
    int b, k, j, i;
public:
    call_dsyr2k_functor( double ***tile_, int b_, int k_, int j_, int i_ )
        : tile(tile_), b(b_), k(k_), j(j_), i(i_) {}

    void operator()( const tbb::flow::continue_msg & ) { call_dsyr2k( tile, b, k, j, i ); }
};

#endif // !__TBB_CPP11_LAMBDAS_PRESENT

class algorithm_depend : public algorithm
{
public:
    algorithm_depend() : algorithm("depend_cholesky", true) {}

protected:
    virtual void func( void * ptr, int n, int b ) {
        double ***tile = (double ***)ptr;

        const int p = n/b;
        continue_ptr_type *c = new continue_ptr_type[p];
        continue_ptr_type **t = new continue_ptr_type *[p];
        continue_ptr_type ***u = new continue_ptr_type **[p];

        tbb::flow::graph g;
        for ( int k = p-1; k >= 0; --k ) {
            c[k] = new continue_type( g,
#if __TBB_CPP11_LAMBDAS_PRESENT
                [=]( const tbb::flow::continue_msg & ) { call_dpotf2( tile, b, k ); } );
#else
                call_dpotf2_functor( tile, b, k ) );
#endif // __TBB_CPP11_LAMBDAS_PRESENT
            t[k] = new continue_ptr_type[p];
            u[k] = new continue_ptr_type *[p];

            for ( int j = k+1; j < p; ++j ) {
                t[k][j] = new continue_type( g,
#if __TBB_CPP11_LAMBDAS_PRESENT
                    [=]( const tbb::flow::continue_msg & ) { call_dtrsm( tile, b, k, j ); } );
#else
                    call_dtrsm_functor( tile, b, k, j ) );
#endif // __TBB_CPP11_LAMBDAS_PRESENT
                make_edge( *c[k], *t[k][j] );
                u[k][j] = new continue_ptr_type[p];

                for ( int i = k+1; i <= j; ++i ) {
                    u[k][j][i] = new continue_type( g,
#if __TBB_CPP11_LAMBDAS_PRESENT
                        [=]( const tbb::flow::continue_msg & ) { call_dsyr2k( tile, b, k, j, i ); } );
#else
                        call_dsyr2k_functor( tile, b, k, j, i ) );
#endif // __TBB_CPP11_LAMBDAS_PRESENT

                    if ( k < p-2 && k+1 != j && k+1 != i ) {
                        make_edge( *u[k][j][i], *u[k+1][j][i] );
                    }

                    make_edge( *t[k][j], *u[k][j][i] );

                    if ( i != j ) {
                        make_edge( *t[k][i], *u[k][j][i] );
                    }

                    if ( k < p-2 && j > i && i == k+1 ) {
                        make_edge( *u[k][j][i], *t[i][j] );
                    }
                }
            }

            if ( k != p-1 ) {
                make_edge( *u[k][k+1][k+1], *c[k+1] );
            }
        }

        c[0]->try_put( tbb::flow::continue_msg() );
        g.wait_for_all();
    }
}; // class algorithm_depend

/************************************************************
 End dependence graph based version of cholesky
************************************************************/

bool process_args( int argc, char *argv[] ) {
    utility::parse_cli_arguments( argc, argv,
        utility::cli_argument_pack()
        //"-h" option for displaying help is present implicitly
        .positional_arg( g_n, "size", "the row/column size of NxN matrix (size <= 46000)" )
        .positional_arg( g_b, "blocksize", "the block size; size must be a multiple of the blocksize" )
        .positional_arg( g_num_trials, "num_trials", "the number of times to run each algorithm" )
        .positional_arg( g_output_prefix, "output_prefix",
            "if provided the prefix will be preappended to output files:\n"
            "                     output_prefix_posdef.txt\n"
            "                     output_prefix_X.txt; where X is the algorithm used\n"
            "                 if output_prefix is not provided, no output will be written" )
        .positional_arg( g_alg_name, "algorithm", "name of the used algorithm - can be dpotrf, crout, depend or join" )
        .positional_arg( g_num_tbb_threads, "num_tbb_threads", "number of started TBB threads" )

        .arg( g_input_file_name, "input_file", "if provided it will be read to get the input matrix" )
        .arg( g_benchmark_run, "-x", "skips all validation" )
    );

    if ( g_n > 46000 ) {
        printf( "ERROR: invalid 'size' value (must be less or equal 46000): %d\n", g_n );
        return false;
    }

    if ( g_n%g_b != 0 ) {
        printf( "ERROR: size %d must be a multiple of the blocksize %d\n", g_n, g_b );
        return false;
    }

    if ( g_n/g_b > 256 ) {
        // Because tile index size is 1 byte only in tag_t type
        printf( "ERROR: size / blocksize must be less or equal 256, but %d / %d = %d\n", g_n, g_b, g_n/g_b );
        return false;
    }

    if ( g_b == -1 || (g_n == -1 && g_input_file_name == NULL) ) {
        return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    typedef std::map< std::string, algorithm * > algmap_t;
    algmap_t algmap;

    // Init algorithms
    algmap.insert(std::pair<std::string, algorithm *>("dpotrf", new algorithm_dpotrf));
    algmap.insert(std::pair<std::string, algorithm *>("crout", new algorithm_crout));
    algmap.insert(std::pair<std::string, algorithm *>("depend", new algorithm_depend));
    algmap.insert(std::pair<std::string, algorithm *>("join", new algorithm_join));

    if ( !process_args( argc, argv ) ) {
        printf( "ERROR: Invalid arguments. Run: %s -h\n", argv[0] );
        exit( 1 );
    }

    tbb::task_scheduler_init init( g_num_tbb_threads );
    double *A = NULL;

    // Read input matrix
    matrix_init( A, g_n, g_input_file_name );

    // Write input matrix if output_prefix is set and we didn't read from a file
    if ( !g_input_file_name && g_output_prefix ) {
        std::string s( g_output_prefix );
        s += "_posdef.txt";
        matrix_write( A, g_n, s.c_str() );
    }

    if ( g_alg_name.empty() ) {
        for ( algmap_t::iterator i = algmap.begin(); i != algmap.end(); ++i ) {
            algorithm* const alg = i->second;
            (*alg)( A, g_n, g_b, g_num_trials );
        }
    }
    else {
        algmap_t::iterator alg_iter = algmap.find(g_alg_name);

        if ( alg_iter != algmap.end() ) {
            algorithm* const alg = alg_iter->second;
            (*alg)( A, g_n, g_b, g_num_trials );
        }
        else {
            printf( "ERROR: Invalid algorithm name: %s\n", g_alg_name.c_str() );
            exit( 2 );
        }
    }

    free( A );
    return 0;
}
