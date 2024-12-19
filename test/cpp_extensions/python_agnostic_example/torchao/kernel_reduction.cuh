//    Copyright 2024 FP6-LLM authors
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
// This file is copied from https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/include/kernel_reduction.cuh

/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
// Used for the reduction of result matrix if Split-K is used
// Reduction_Workspace:     (Split_K, M_Global, N_Global),  column major
// C:                       (M_Global, N_Global),           column major
// Each thread deals with 8 output elements, each elements is the sum of Split_K elements
//      Read  Global: Each Warp/ThreadBlock: 32 threads_per_warp * 8 float_per_thread (256bit) -> 256 float per warp
//      Write Global: Each Warp/ThreadBlock: 32 threads_per_warp * 8 half_per_thread  (128bit) -> 256 half  per warp
// GridSize = (M_Global*N_Global) / 256

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define REDUCTION_ELEMENT_PER_THREADBLOCK   256
#define HALF_PER_128BIT                     8

__global__ void SplitK_Reduction(half* C, float* Reduction_Workspace, size_t M_Global, size_t N_Global, int Split_K)
{
    half*  WARP_GPTR_C      = C                    + REDUCTION_ELEMENT_PER_THREADBLOCK * blockIdx.x;
    float* WARP_GPTR_R      = Reduction_Workspace  + REDUCTION_ELEMENT_PER_THREADBLOCK * blockIdx.x;
    half*  THREAD_GPTR_C    = WARP_GPTR_C          + threadIdx.x * HALF_PER_128BIT;
    float* THREAD_GPTR_R    = WARP_GPTR_R          + threadIdx.x * HALF_PER_128BIT;
    // Initializing Thread-Local Results
    float Results[HALF_PER_128BIT];
    #pragma unroll
    for (int i = 0; i < HALF_PER_128BIT; i++)       Results[i] = 0.0f;
    // Reduction
    for (int i = 0; i < Split_K; i++) {
        #pragma unroll
        for (int j = 0; j < HALF_PER_128BIT; j++)   Results[j] += THREAD_GPTR_R[j];
        THREAD_GPTR_R += M_Global * N_Global;
    }
    // Writing to global memory
    #pragma unroll
    for (int i = 0; i < HALF_PER_128BIT; i++)       THREAD_GPTR_C[i] = __float2half_rn(Results[i]);
}
