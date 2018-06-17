/*
    Copyright (c) 2016-2018 Intel Corporation

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

// -------------------------------------------------------------------------------------------------
// This is an example of a matrix multiplication Gen kernel usage with TBB Flow Graph.
// It exemplifies support for compute offload to Intel(R) Graphics Technology in the flow graph API.
// -------------------------------------------------------------------------------------------------

#define TBB_PREVIEW_FLOW_GRAPH_NODES 1
#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1

#include "tbb/tbb_config.h"
#include "../../common/utility/utility.h"

#if __TBB_PREVIEW_GFX_FACTORY && __TBB_PREVIEW_STREAMING_NODE

#if _MSC_VER
#pragma warning(disable : 4503) // suppress warning C4503: decorated name length exceeded, name was truncated 
#endif

// -------------------------------------------------------------------------------------------------

#include <iostream>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include "tbb/tick_count.h"
#include "tbb/flow_graph.h"
#include "tbb/gfx_factory.h"

// -------------------------------------------------------------------------------------------------

#define SLM_TILE_X (4*8)
#define SLM_TILE_Y (4*8)
#define SLM_TILE_K (4*8)

#define SIZE_X (2*SLM_TILE_X)
#define SIZE_Y (2*SLM_TILE_Y)
#define SIZE_K (2*SLM_TILE_X)

#ifdef __GFX__
#define BARRIER _gfx_gpgpu_thread_barrier()
#else
#define BARRIER
#endif

#define TILE_Y 8
#define TILE_K 8
#define TILE_X 8

// The naive straightforward algorithm used to obtain reference results on CPU
void matmult_naive(const float* A, const float* B, float* C) {
    for (int y = 0; y < SIZE_Y; y++) {
        for (int x = 0; x < SIZE_X; x++) {
            C[y * SIZE_Y + x] = (float)0;

            for (int k = 0; k < SIZE_K; k++) {
                C[y * SIZE_Y + x] += A[y * SIZE_K + k] * B[k * SIZE_K + x];
            }
        }
    }
}

// Shared Local Memory based algorithm
__declspec(target(gfx_kernel))
void matmult_tiled_slm(const float A[][SIZE_K], const float B[][SIZE_X], float C[][SIZE_X]) {
    // The parallel loop nest below iterates over "supertiles" in the resulting
    // matrix C and it is parallelized across thread groups, 1 iteration per
    // group, which effectively means that the loop nest is peeled off.
    // This kernel is programmed so that each thread group calculates one
    // resulting supertile in matrix C.
    _Cilk_for _Thread_group(int tg_y = 0; tg_y < SIZE_Y; tg_y += SLM_TILE_Y) {
        _Cilk_for _Thread_group(int tg_x = 0; tg_x < SIZE_X; tg_x += SLM_TILE_X) {
            // declare "supertiles" of each matrix to be allocated in SLM
            __thread_group_local float slm_atile[SLM_TILE_Y][SLM_TILE_K];
            __thread_group_local float slm_btile[SLM_TILE_K][SLM_TILE_X];
            __thread_group_local float slm_ctile[SLM_TILE_Y][SLM_TILE_X];

            // initialize the result supertile (in parallel)
            //slm_ctile[:][:] = (float)0;
            _Cilk_for(int i0 = 0; i0 < SLM_TILE_Y; i0++)
                _Cilk_for(int i1 = 0; i1 < SLM_TILE_X; i1++)
                    slm_ctile[i0][i1] = (float)0;

            // calculate the dot product of supertiles:
            for (int super_k = 0; super_k < SIZE_K; super_k += SLM_TILE_K) {
                // cache A's and B's "supertiles" in SLM (in parallel)
                //slm_atile[:][:] = A[tg_y:SLM_TILE_Y][super_k:SLM_TILE_K];
                _Cilk_for(int i0 = 0; i0 < SLM_TILE_Y; i0++)
                    _Cilk_for(int i1 = 0; i1 < SLM_TILE_K; i1++)
                        slm_atile[i0][i1] = A[tg_y + i0][super_k + i1];

                //slm_btile[:][:] = B[super_k:SLM_TILE_K][tg_x:SLM_TILE_X];
                _Cilk_for(int i0 = 0; i0 < SLM_TILE_K; i0++)
                    _Cilk_for(int i1 = 0; i1 < SLM_TILE_X; i1++)
                        slm_btile[i0][i1] = B[super_k + i0][tg_x + i1];

                // need a barrier, since every tile in tiles are used by
                // multiple threads in the group
                BARRIER;

                // now multiply the supertiles as usual matrices (in parallel)
                // ...
                // ... using the most effective tiled algorithm:
                _Cilk_for(int t_y = 0; t_y < SLM_TILE_Y; t_y += TILE_Y) {
                    _Cilk_for(int t_x = 0; t_x < SLM_TILE_X; t_x += TILE_X) {
                        // allocate tiles in registers
                        float atile[TILE_Y][TILE_K], btile[TILE_X];
                        float ctile[TILE_Y][TILE_X];

                        // ... and initialize ctile to zero
                        ctile[:][:] = (float)0;

                        // calculate the dot product of the tiles
                        for (int k = 0; k < SLM_TILE_K; k += TILE_K) {
                            atile[:][:] = slm_atile[t_y:TILE_Y][k:TILE_K];

                            for (int k_ind = 0; k_ind < TILE_K; k_ind++) {
                                btile[:] = slm_btile[k + k_ind][t_x:TILE_X];

                                // multiply current btile row by atile's
                                // current element and add up to corresponding
                                // ctile row
                                for (int y_ind = 0; y_ind < TILE_Y; y_ind++) {
                                    ctile[y_ind][:] += atile[y_ind][k_ind] *
                                        btile[:];
                                }
                            }
                        }
                        // flush the thread-local ctile (registers) into the
                        // thread group-local supertile (SLM) adding up
                        // elements
                        slm_ctile[t_y:TILE_Y][t_x:TILE_X] += ctile[:][:];
                    }
                }

                // barrier to make sure
                // (1) next iteration of the loop does not overwrite a and b
                //   SLM tiles used in the above calculation of slm_ctile
                // (2) on the last iteration of the loop, all threads wait
                //   for the SLM ctile calculation to be completed before
                //   writing it back to memory below this loop
                BARRIER;
            }

            // write (in parallel) the result supertile back to memory:
            //C[tg_y:SLM_TILE_Y][tg_x:SLM_TILE_X] = slm_ctile[:][:];
            _Cilk_for(int i0 = 0; i0 < SLM_TILE_Y; i0++)
                _Cilk_for(int i1 = 0; i1 < SLM_TILE_X; i1++)
                    C[tg_y + i0][tg_x + i1] = slm_ctile[i0][i1];

            // next iteration of the loop zeroes out slm_ctile - make sure this
            // always happens after slm_ctile has been dumped to memory (above
            // loop nest completes):
            BARRIER;
        }
    }
}

// Matrix initialization function
void init_matrix(int width, int height, float* matrix) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            matrix[j * height + i] = (float)(j * 4 + i);
        }
    }
}

// Validate matrix with reference values
bool validate_mat(char *title, int size_y, int size_x, const float* mat, const float* ref_mat) {
    int err_cnt = 0;
    printf("verifying %s...", title);
    fflush(stdout);

    for (int y = 0; y < size_y; y++) {
        for (int x = 0; x < size_x; x++) {
            float val = mat[y * size_y + x];
            float gold_val = ref_mat[y * size_y + x];

            if (val != gold_val) {
                if (err_cnt == 0) {
                    printf("\n");
                }
                if (++err_cnt < 10) {
                    std::cout <<
                        "  ERROR at [" << y << "][" << x << "]: " <<
                        val << "(should be " << gold_val << ")" << std::endl;
                }
            }
        }
    }
    if (err_cnt == 0) {
        printf(" ok\n");
    } else {
        printf("FAILED\n");
    }
    return err_cnt == 0;
}

// ----------------------------------------------------------------------------
// Helper types and functions
// ----------------------------------------------------------------------------

struct matrix_size {
    int columns;
    int rows;
};

typedef std::tuple< tbb::flow::gfx_buffer<float>,
                    tbb::flow::gfx_buffer<float>,
                    tbb::flow::gfx_buffer<float> > kernel_args_type;
typedef kernel_args_type cpu_args_type;
typedef kernel_args_type validation_args_type;

// Constructs flow graph with three computation nodes that all make matrixes multiplication
// - CPU node - obtains reference result on CPU
// - CPU SLM node - Intel(R) Cilk(TM) based matrixes multiplication implementation on CPU
// - GPU node - obtains result on GPU using GFX offload API
void mat_multiplication() {

    //------------------------------------------
    // TBB Flow Graph nodes declaration section
    //------------------------------------------

    tbb::flow::graph g;
    tbb::flow::gfx_factory factory(g);

    // Enqueue task for running on Gen
    tbb::flow::split_node< kernel_args_type > gpu_slm_split_n(g);
    tbb::flow::streaming_node< kernel_args_type, tbb::flow::queueing, tbb::flow::gfx_factory > gpu_slm_mat_mult_n(g, matmult_tiled_slm, tbb::flow::gfx_factory::dummy_device_selector(), factory);

    // Obtain SLM algorithm result on CPU
    tbb::flow::function_node< cpu_args_type, tbb::flow::gfx_buffer<float> > cpu_slm_mat_mult_n(g, tbb::flow::unlimited, [](const cpu_args_type& args) -> tbb::flow::gfx_buffer<float> {
        // Get references to matrixes
        const tbb::flow::gfx_buffer<float >& A_MATRIX = std::get<0>(args);
        const tbb::flow::gfx_buffer<float>& B_MATRIX  = std::get<1>(args);
        tbb::flow::gfx_buffer<float> CPU_SLM_MATRIX   = std::get<2>(args);

        matmult_tiled_slm((float(*)[SIZE_K])A_MATRIX.data(), (float(*)[SIZE_X])B_MATRIX.data(), (float(*)[SIZE_X])CPU_SLM_MATRIX.data());

        return CPU_SLM_MATRIX;
    });

    // Obtain reference result on CPU
    tbb::flow::function_node< cpu_args_type, tbb::flow::gfx_buffer<float> > cpu_naive_mat_mult_n(g, tbb::flow::unlimited, [](const cpu_args_type& args) -> tbb::flow::gfx_buffer<float> {
        // Get references to matrixes
        const tbb::flow::gfx_buffer<float>& A_MATRIX  = std::get<0>(args);
        const tbb::flow::gfx_buffer<float>& B_MATRIX  = std::get<1>(args);
        tbb::flow::gfx_buffer<float> CPU_NAIVE_MATRIX = std::get<2>(args);

        matmult_naive(A_MATRIX.data(), B_MATRIX.data(), CPU_NAIVE_MATRIX.data());

        return CPU_NAIVE_MATRIX;
    });

    // Validate computed matrixes
    tbb::flow::join_node< validation_args_type > validation_join_n(g);
    tbb::flow::function_node< validation_args_type > mat_validation_n(g, tbb::flow::unlimited, [](const validation_args_type& result) {
        // Get references to matrixes
        const tbb::flow::gfx_buffer<float>& GPU_SLM_MAT   = std::get<0>(result);
        const tbb::flow::gfx_buffer<float>& CPU_SLM_MAT   = std::get<1>(result);
        const tbb::flow::gfx_buffer<float>& CPU_NAIVE_MAT = std::get<2>(result);

        // Verify results
        // Check that slm algorithm produces correct results on CPU:
        validate_mat("matrix multiply: 'SLM' CPU vs. CPU", SIZE_Y, SIZE_X, CPU_SLM_MAT.data(), CPU_NAIVE_MAT.data());
        // Verify Gen results:
        validate_mat("matrix multiply: SLM Gen vs. CPU", SIZE_Y, SIZE_X, GPU_SLM_MAT.data(), CPU_NAIVE_MAT.data());
    });

    //-----------------------------------------
    // Make edge section - connecting nodes
    //-----------------------------------------

    // Prepare main graph input ports for data
    make_edge(tbb::flow::output_port<0>(gpu_slm_split_n), tbb::flow::input_port<0>(gpu_slm_mat_mult_n));
    make_edge(tbb::flow::output_port<1>(gpu_slm_split_n), tbb::flow::input_port<1>(gpu_slm_mat_mult_n));
    make_edge(tbb::flow::output_port<2>(gpu_slm_split_n), tbb::flow::input_port<2>(gpu_slm_mat_mult_n));

    // Join results
    make_edge(tbb::flow::output_port<2>(gpu_slm_mat_mult_n), tbb::flow::input_port<0>(validation_join_n));
    make_edge(cpu_slm_mat_mult_n, tbb::flow::input_port<1>(validation_join_n));
    make_edge(cpu_naive_mat_mult_n, tbb::flow::input_port<2>(validation_join_n));

    //Verify correctness
    make_edge(validation_join_n, mat_validation_n);

    // Set args for GFX kernel.
    // Default behaviour if not set.
    gpu_slm_mat_mult_n.set_args(tbb::flow::port_ref<0, 2>);

    //-----------------------------------------
    // Input sizes and matrixes initialization
    //-----------------------------------------

    const matrix_size A_MATRIX_SIZE         = { SIZE_Y, SIZE_K };
    const matrix_size B_MATRIX_SIZE         = { SIZE_K, SIZE_X };
    const matrix_size GPU_SLM_MATRIX_SIZE   = { SIZE_Y, SIZE_X };
    const matrix_size CPU_SLM_MATRIX_SIZE   = { SIZE_Y, SIZE_X };
    const matrix_size CPU_NAIVE_MATRIX_SIZE = { SIZE_Y, SIZE_X };

    tbb::flow::gfx_buffer<float> A_MATRIX(A_MATRIX_SIZE.columns * A_MATRIX_SIZE.rows);
    tbb::flow::gfx_buffer<float> B_MATRIX(B_MATRIX_SIZE.columns * B_MATRIX_SIZE.rows);
    tbb::flow::gfx_buffer<float> GPU_SLM_MATRIX(GPU_SLM_MATRIX_SIZE.columns * GPU_SLM_MATRIX_SIZE.rows);
    tbb::flow::gfx_buffer<float> CPU_SLM_MATRIX(CPU_SLM_MATRIX_SIZE.columns * CPU_SLM_MATRIX_SIZE.rows);
    tbb::flow::gfx_buffer<float> CPU_NAIVE_MATRIX(CPU_NAIVE_MATRIX_SIZE.columns * CPU_NAIVE_MATRIX_SIZE.rows);

    // Intitialize input matrixes
    init_matrix(A_MATRIX_SIZE.columns, A_MATRIX_SIZE.rows, A_MATRIX.data());
    init_matrix(B_MATRIX_SIZE.columns, B_MATRIX_SIZE.rows, B_MATRIX.data());

    // Make tuples with input data for graph
    kernel_args_type GPU_SLM_INPUT   = std::make_tuple(A_MATRIX, B_MATRIX, GPU_SLM_MATRIX);
    kernel_args_type CPU_SLM_INPUT   = std::make_tuple(A_MATRIX, B_MATRIX, CPU_SLM_MATRIX);
    kernel_args_type CPU_NAIVE_INPUT = std::make_tuple(A_MATRIX, B_MATRIX, CPU_NAIVE_MATRIX);

    //-----------------------------------------
    // Send input to the graph and run it
    //-----------------------------------------

    gpu_slm_split_n.try_put(GPU_SLM_INPUT);
    cpu_slm_mat_mult_n.try_put(CPU_SLM_INPUT);
    cpu_naive_mat_mult_n.try_put(CPU_NAIVE_INPUT);

    // Run graph
    g.wait_for_all();
}

//---------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    try {
        tbb::tick_count mainStartTime = tbb::tick_count::now();

        utility::parse_cli_arguments(argc, argv,
            utility::cli_argument_pack()
            //"-h" option for displaying help is present implicitly
        );

        // Compute matrices and verify result
        mat_multiplication();

        utility::report_elapsed_time((tbb::tick_count::now() - mainStartTime).seconds());
        return 0;
    } catch (std::exception& e) {
        std::cerr << "Error occurred. Error text is : \"" << e.what() << "\"\n";
        return -1;
    }
}

#else
int main() {
    utility::report_skipped();
    return 0;
}
#endif /* __TBB_PREVIEW_GFX_FACTORY && __TBB_PREVIEW_STREAMING_NODE */
