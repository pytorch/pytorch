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
// This file is modified from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/include/ptx_mma.cuh
//
// MODIFICATION NOTE (2024-09-25): added SM75 support (https://github.com/pytorch/ao/pull/942):
// - Replaced m16n8k16 Tensor core operation with two m16n8k8 operations
// - Accounted for a difference in expected parameters for the ldmatrix operation

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
#ifndef PTX_MMA_CUH
#define PTX_MMA_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <assert.h>
#include "configs.h"

// MODIFICATION NOTE: to support MSVC
// - uint32_t __restrict__ Reg[][4] is changed to uint32_t (* __restrict__ Reg)[4]
// - half __restrict__ (*read_SPTR) is changed to half (* __restrict__ read_SPTR)
template <typename TilingConfig>
__device__ __forceinline__ void B_FromSharedToReg(uint32_t (* __restrict__ Reg)[4],
                                                  half     (* __restrict__ read_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                  int                      slice_id) {
    #ifdef DEBUG_MODE
        static_assert( (TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0) );
    #endif

    const int   warpId  = threadIdx.x / WARP_SIZE;
    int         lane_id = threadIdx.x % WARP_SIZE;
    int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_col = TilingConfig::WARP_COL_MMA_TENSORS * MMA_8 * WARP_j;   // each warp may start from reading warp_start_col'th column of the B tile in shared memory
    #ifdef DEBUG_MODE
        assert( warp_start_col==0 );
    #endif

    #if __CUDA_ARCH__ == 750
    if (TilingConfig::WARP_COL_MMA_TENSORS==1) {
      // For .target sm_75, all threads must contain valid addresses for the 'ldmatrix' op. below. Otherwise, the behavior is undefined.
      // See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix
      // To avoid this, we make threads 16-32 point to the same smem addresses as threads 0-15 by changing the lane id.
      lane_id = lane_id % 16;
    }
    #endif
    int col = (lane_id%8) + (lane_id/16)*8;
    int row = (lane_id%16) / 8 * 8;
    uint32_t smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&read_SPTR[warp_start_col+col][slice_id*MMA_16 + row]));
    if(TilingConfig::WARP_COL_MMA_TENSORS==1) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(Reg[0][0]), "=r"(Reg[0][1])
                     : "r"(smem_local_ptr));
    }
    else {
        #pragma unroll
        for (int i = 0; i < TilingConfig::WARP_COL_MMA_TENSORS/2; i++)
        {
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(Reg[i][0]), "=r"(Reg[i][1]), "=r"(Reg[i][2]), "=r"(Reg[i][3])
                         : "r"(smem_local_ptr));
            smem_local_ptr += 16 * (WARP_K+PADDING_SHARED_MEM_FOR_B_8) * sizeof(half);
        }
    }
}

// MODIFICATION NOTE: to support MSVC, the function signature is changed from
// MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b).
__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t * __restrict__ c, uint32_t * __restrict__ a, uint32_t * __restrict__ b)
{
  #if __CUDA_ARCH__ == 750
    // m16n8k16 op. requires >=sm_80, so instead we use two m16n8k8 ops.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5},"
                 "{ %6 },"
                 "{ %7, %8, %9, %10 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]),
                   "r"(b[0]),
                   "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5},"
                 "{ %6 },"
                 "{ %7, %8, %9, %10 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[2]), "r"(a[3]),
                   "r"(b[1]),
                   "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

  #else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]),
                   "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
  #endif
}

#endif
