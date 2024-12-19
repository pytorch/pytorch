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
// This file is modified from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/include/utils_core.cuh

#ifndef UTILS_CORE_CUH
#define UTILS_CORE_CUH

#include <assert.h>

#include "configs.h"
#include "ptx_mma.cuh"
#include "utils_parallel_dequant.cuh"


template<int NUM_INT_PER_THREAD>
__device__ __forceinline__ void CopyFromSharedToRegister_AFrag(uint32_t Reg[], uint32_t* SPTR, int slice_id) {
    SPTR += slice_id * (NUM_INT_PER_THREAD*WARP_SIZE);
    int     lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for(int i=0; i<NUM_INT_PER_THREAD; i++) {
        Reg[i] = SPTR[lane_id+i*WARP_SIZE];
    }
}

// MODIFICATION NOTE: to support MSVC, half __restrict__ (*B_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] is changed to below.
template <typename TilingConfig, int EXPONENT, int MANTISSA>
__device__ __forceinline__ void initialize_mma_slice(uint32_t                  (*a)[4],
                                                     uint32_t                  (*b)[4],
                                                     uint32_t* __restrict__    A_1BIT_SPTR_read,
                                                     uint32_t* __restrict__    A_2BIT_SPTR_read,
                                                     uint32_t* __restrict__    A_4BIT_SPTR_read,
                                                     half   (* __restrict__    B_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                     uint32_t*                 RPTR_Scales)
{
    // 1+2+4 weight split
    constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread;
    uint32_t a_1bit[1];                      // NO double buffer
    uint32_t a_2bit[2];                      // NO double buffer
    uint32_t a_4bit[4];                      // NO double buffer
    if(USE_SEG_1BIT) CopyFromSharedToRegister_AFrag<1>   (a_1bit, A_1BIT_SPTR_read, 0);
    if(USE_SEG_2BIT) CopyFromSharedToRegister_AFrag<2>   (a_2bit, A_2BIT_SPTR_read, 0);
    if(USE_SEG_4BIT) CopyFromSharedToRegister_AFrag<4>   (a_4bit, A_4BIT_SPTR_read, 0);
    Dequant_32FP6_4Way<EXPONENT, MANTISSA>(a, a_1bit, a_2bit, a_4bit, RPTR_Scales);   // SIMT Dequant: dequantizing FPx to FP16 at register level, dequantizing a slice each time
    B_FromSharedToReg<TilingConfig>(b, B_SPTR_read, 0); // Loading B from shared to registers
}

// MODIFICATION NOTE: to support MSVC, half __restrict__ (*B_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] is changed to below.
template <typename TilingConfig, int EXPONENT, int MANTISSA>
__device__ __forceinline__ void core_mma_slice(float                     c[][REG_PER_THREAD_C_TENSOR_16_16],
                                               uint32_t                  (*a)[4],
                                               uint32_t                  (*b)[4],
                                               uint32_t* __restrict__    A_1bit_SPTR_read,
                                               uint32_t* __restrict__    A_2bit_SPTR_read,
                                               uint32_t* __restrict__    A_4bit_SPTR_read,
                                               half   (* __restrict__    B_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                               uint32_t*                 RPTR_Scales,
                                               int                       slice_id)      // writing slice[slice_id] to registers, k=0 -> slice_id=1 for prefetching
{
    // 1+2+4 weight split
    constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
    constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
    constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
    constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;

    #ifdef DEBUG_MODE
        assert((TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0));   // if WARP_COL_MMA_TENSORS == 1, B tile in registers is padded to a 16*16 MMA block
    #endif
    const int NumRegSets_a = WARP_ROW_MMA_TENSORS;                                                                              // 1 set = 4 registers, containing a 16*16 MMA block
    const int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;                // 1 set = 4 registers, containing a 16*16 MMA block
    uint32_t (*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(c);    // Reigsters for accumulated FP32 results

    // Setting RPTRs for double buffers
    uint32_t (*a_read )[4] = a;
    uint32_t (*a_write)[4] = a;
    uint32_t (*b_read )[4] = b;
    uint32_t (*b_write)[4] = b;
    if(slice_id%2==1)   { b_write += NumRegSets_b; a_write += NumRegSets_a;}
    else                { b_read  += NumRegSets_b; a_read  += NumRegSets_a;}

    // Reading registers and issuing core tensor core computations (a slice of A and B tile in shared memory)
    #pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
        if(TilingConfig::WARP_COL_MMA_TENSORS==1) {
            MMA_FP16_M16N8K16( c_uint_ptr[i], a_read[i], b_read[0] );
        }
        else {
            #pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS/2; j++) {
                MMA_FP16_M16N8K16( c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS],     a_read[i], b_read[j]     );
                MMA_FP16_M16N8K16( c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS] + 4, a_read[i], b_read[j] + 2 ); // c+4; b+2
            }
        }
    }
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread;
    uint32_t a_1bit[1];                      // NO double buffer
    uint32_t a_2bit[2];                      // NO double buffer
    uint32_t a_4bit[4];                      // NO double buffer
    if(USE_SEG_1BIT) CopyFromSharedToRegister_AFrag<1>   (a_1bit, A_1bit_SPTR_read, slice_id);
    if(USE_SEG_2BIT) CopyFromSharedToRegister_AFrag<2>   (a_2bit, A_2bit_SPTR_read, slice_id);
    if(USE_SEG_4BIT) CopyFromSharedToRegister_AFrag<4>   (a_4bit, A_4bit_SPTR_read, slice_id);
    Dequant_32FP6_4Way<EXPONENT, MANTISSA>(a_write, a_1bit, a_2bit, a_4bit, RPTR_Scales);   // SIMT Dequant: dequantizing FP6 to FP16 at register level, dequantizing a slice each time
    B_FromSharedToReg<TilingConfig>     (b_write, B_SPTR_read, slice_id); // Loading B from shared to registers
}

template <typename TilingConfig>
__device__ __forceinline__ void StoreToSharedMemoryFromRegister(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4],
                                                                float c[][REG_PER_THREAD_C_TENSOR_16_16])
{
    const int   lane_id             = threadIdx.x % WARP_SIZE;
    const int   warpId              = threadIdx.x / WARP_SIZE;
    int         warp_row_offset     = warpId * (MMA_16 * WARP_ROW_MMA_TENSORS);
    #pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
        #pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS; j++) {    // Dealing with one 16*8 Tensor
            int RegSetID            = i + (j/2)*WARP_ROW_MMA_TENSORS;
            int RegOffset           = (j%2)*(REG_PER_THREAD_C_TENSOR_16_16/2);
            int Tensor_row_offset   = warp_row_offset + i * MMA_16;
            int Tensor_col_offset   = j * MMA_8;
            #pragma unroll
            for (int r = 0; r < REG_PER_THREAD_C_TENSOR_16_16/2; r++) {
                int row_offset = lane_id / 4;
                if (r >= 2) row_offset += 8;
                int col_offset = (lane_id % 4) * 2;
                if (r%2==1) col_offset += 1;
                smem_CFrag[Tensor_col_offset + col_offset][Tensor_row_offset + row_offset] = c[RegSetID][r + RegOffset];
            }
        }
    }
}

#endif
