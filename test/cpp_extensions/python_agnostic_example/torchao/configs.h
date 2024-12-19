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
// This file is copied from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/include/configs.h

#ifndef CONFIGS_H
#define CONFIGS_H

//#define DEBUG_MODE
#define PIPELINE_LEVEL_GMEM 2
#define PIPELINE_LEVEL_SMEM 2       // only support 2

/************************ Hardware Parameters ************************/
#define WARP_SIZE                           32
#define REG_BIT_WIDTH                       32
// mma: M=16 K=16 N=8
#define MMA_8                               8
#define MMA_16                              16
// for memory access
#define THREAD_OPT_ACCESS_BIT_WIDTH_128     128 // LDS.128, cp_async.128, ...
#define BIT_WIDTH_PER_HALF                  16  // Half precision: FP16

/******************** Register Allocation For GEMM ********************/
#define REG_PER_THREAD_C_TENSOR_16_16       8   // 8 for FP32 Accumulation
/********************** Memory Padding Parameters **********************/
// Eliminating bank-conflict
#define PADDING_BYTES_16                    16 // Padding 16 bytes each column
#define PADDING_SHARED_MEM_FOR_B_8          8  // Padding 8 half  each column, during CopyFromGlobalToShared() for B
#define PADDING_SHARED_MEM_FOR_C_4          4  // Padding 4 float each column, during StoreToSharedMemoryFromRegister() for C
/************************* WARP Tiling part-1 *************************/
#define WARP_ROW_MMA_TENSORS                4
#define WARP_M                              (WARP_ROW_MMA_TENSORS * MMA_16)       // 64
#define WARP_K_MMA_TENSORS                  4
#define WARP_K                              (WARP_K_MMA_TENSORS   * MMA_16)       // 64
template<int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_MMA_TENSORS_>
struct TilingConfig {
    // Depending on "n" dimension of the GEMM
    static constexpr int BLOCK_ROW_WARPS        = BLOCK_ROW_WARPS_;
    static constexpr int BLOCK_COL_WARPS        = BLOCK_COL_WARPS_;
    static constexpr int WARP_COL_MMA_TENSORS   = WARP_COL_MMA_TENSORS_;
    /************************* WARP Tiling part-2 *************************/
    static constexpr int WARP_N                 = WARP_COL_MMA_TENSORS * MMA_8;
    /*************************Thread Block Tiling *************************/
    static constexpr int TILE_M                 = WARP_M * BLOCK_ROW_WARPS;
    static constexpr int TILE_N                 = MMA_8  * WARP_COL_MMA_TENSORS * BLOCK_COL_WARPS;
    static constexpr int TILE_K                 = WARP_K;
    /********************** #Thread per Thread Block **********************/
    static constexpr int BLOCK_WARPS   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
    /******************************* Others *******************************/
    static constexpr int SMEM_SIZE_B_TILE   = TILE_N * (TILE_K + PADDING_BYTES_16) * 2 * PIPELINE_LEVEL_GMEM;          // sizeof(half)=2, doubleBuffer=2
    static constexpr int SMEM_SIZE_C_TILE   = TILE_N * (TILE_M + PADDING_BYTES_16) * 4;                             // sizeof(float)=4
};



#endif  // CONFIGS_H
