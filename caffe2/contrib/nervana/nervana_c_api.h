/*
 * Copyright 2015 Baidu USA, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/** Load all the sgemm and hgemm cubins from the given path
 * \param [in] base_path path to the kernel cubins
 * \return true on success and false if an error was encountered
 */
bool nervana_loadKernels(const char* const base_path);

/** Unload all currently loaded cubins
 * \return true on success and false if an error was encountered
 */
bool nervana_unloadKernels();

/** Return the number of bytes required for the random state
 *  used in stochastic rounding.
 *  \return bytes required for random state
 */
 size_t nervana_randStateSizeBytes();

/** Perform BLAS sgemm on alpha * A * B + beta * C, with the
 *  additional options of stochastic rounding and applying a
 *  rectified linear unit (relu) to the result.  This routine expects
 *  all matrices to be in row-major order.
 *  \param [in] A Pointer to the data for matrix A
 *  \param [in] B Pointer to the data for matrix B
 *  \param [in, out] C Pointer to the data for matrix C
 *  \param [in] m number of rows of C
 *  \param [in] n number of columns of C
 *  \param [in] k inner dimension of multiplication
 *  \param [in] lda leading dimension of two-dimensional array A
 *  \param [in] ldb leading dimension of two-dimensional array B
 *  \param [in] ldc leading dimension of two-dimensional array C
 *  \param [in] alpha scalar used for multiplication
 *  \param [in] beta scalar used for multiplication
 *  \param [in, out] rand_state pointer to memory used for random state
 *              use nervana_randStateSizeBytes to allocate the correct size
 *              if stochastic_round is false, this can be NULL
 *  \param [in] stochastic_round true if stochastic rounding should be used
 *  \param [in] apply_relu true if a relu should be applied to the result
 *  \param [in] stream The cudaStream on which the kernel should be launched
 *  \param [in] grid Choose a specific grid configuration: 0=32x128, 1=128x32, 2=128x64, 3=128x128
 */
 bool nervana_sgemm(float *A, float *B, float *C,
                    bool a_t, bool b_t,
                    int m, int n, int k,
                    int lda, int ldb, int ldc,
                    float alpha, float beta,
                    unsigned int *rand_state,
                    bool stochastic_round, bool apply_relu,
                    CUstream stream, int grid=-1
                    );

/** Perform BLAS hgemm on alpha * A * B + beta * C, with the
 *  additional options of stochastic rounding and applying a
 *  rectified linear unit (relu) to the result.  This routine expects
 *  all matrices to be in row-major order.
 *  \param [in] A Pointer to the data for matrix A
 *  \param [in] B Pointer to the data for matrix B
 *  \param [in, out] C Pointer to the data for matrix C
 *  \param [in] m number of rows of C
 *  \param [in] n number of columns of C
 *  \param [in] k inner dimension of multiplication
 *  \param [in] lda leading dimension of two-dimensional array A
 *  \param [in] ldb leading dimension of two-dimensional array B
 *  \param [in] ldc leading dimension of two-dimensional array C
 *  \param [in] alpha scalar used for multiplication
 *  \param [in] beta scalar used for multiplication
 *  \param [in, out] rand_state pointer to memory used for random state
 *              use nervana_randStateSizeBytes to allocate the correct size
 *              if stochastic_round is false, this can be NULL
 *  \param [in] stochastic_round true if stochastic rounding should be used
 *  \param [in] apply_relu true if a relu should be applied to the result
 *  \param [in] stream The cudaStream on which the kernel should be launched
 *  \param [in] grid Choose a specific grid configuration: 0=32x128, 1=128x32, 2=128x64, 3=128x128
 */
 bool nervana_hgemm(short *A, short *B, short *C,
                    bool a_t, bool b_t,
                    int m, int n, int k,
                    int lda, int ldb, int ldc,
                    float alpha, float beta,
                    unsigned int *rand_state,
                    bool stochastic_round, bool apply_relu,
                    CUstream stream, int grid=-1
                    );

 bool nervana_sgemm_colmajor(float *A, float *B, float *C,
                             bool a_t, bool b_t,
                             int m, int n, int k,
                             int lda, int ldb, int ldc,
                             float alpha, float beta,
                             unsigned int *rand_state,
                             bool stochastic_round, bool apply_relu,
                             CUstream stream, int grid=-1
                             );

 bool nervana_hgemm_colmajor(short *A, short *B, short *C,
                             bool a_t, bool b_t,
                             int m, int n, int k,
                             int lda, int ldb, int ldc,
                             float alpha, float beta,
                             unsigned int *rand_state,
                             bool stochastic_round, bool apply_relu,
                             CUstream stream, int grid=-1
                             );

#ifdef __cplusplus
}
#endif
