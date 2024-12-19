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
// This file is modified from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/include/kernel_matmul.cuh
//
// MODIFICATION NOTE (2024-09-25): added SM75 support (https://github.com/pytorch/ao/pull/942):
// - Added __CUDA_ARCH__ guards such that async operations are only executed for SM80 and up
//

#include "configs.h"
#include "utils_gmem.cuh"
#include "utils_core.cuh"

/************************** Bitwidth of Weight Segments ************************/
#define BIT_WIDTH_1                 1
#define BIT_WIDTH_2                 2
#define BIT_WIDTH_4                 4
/*************************** 64*64 Weghts of Weight Matrix *********************/
#define WEIGHT_PER_WARP            (WARP_M*WARP_K)                                                            // 64*64 = 4096
#define SMEM_SIZE_PER_WARP_1BIT    (WEIGHT_PER_WARP*BIT_WIDTH_1/8)                                            // 512 Bytes,  doubleBuffer not taken into consideration
#define SMEM_SIZE_PER_WARP_2BIT    (WEIGHT_PER_WARP*BIT_WIDTH_2/8)                                            // 1024 Bytes, doubleBuffer not taken into consideration
#define SMEM_SIZE_PER_WARP_4BIT    (WEIGHT_PER_WARP*BIT_WIDTH_4/8)                                            // 2048 Bytes, doubleBuffer not taken into consideration
#define SMEM_SIZE_PER_TB_1BIT      (SMEM_SIZE_PER_WARP_1BIT*TilingConfig::BLOCK_WARPS*PIPELINE_LEVEL_GMEM)    // #WARP=4; Trible-Buffer for 3-level pipeline for A = 6 KB;  double buffer for 2-level pipeline A= 4  KB.
#define SMEM_SIZE_PER_TB_2BIT      (SMEM_SIZE_PER_WARP_2BIT*TilingConfig::BLOCK_WARPS*PIPELINE_LEVEL_GMEM)    // #WARP=4; Trible-Buffer for 3-level pipeline for A = 12 KB; double buffer for 2-level pipeline A= 8  KB.
#define SMEM_SIZE_PER_TB_4BIT      (SMEM_SIZE_PER_WARP_4BIT*TilingConfig::BLOCK_WARPS*PIPELINE_LEVEL_GMEM)    // #WARP=4; Trible-Buffer for 3-level pipeline for A = 24 KB; double buffer for 2-level pipeline A= 16 KB.
#define SMEM_SIZE_PER_TB_A_TILE    (SMEM_SIZE_PER_TB_1BIT+SMEM_SIZE_PER_TB_2BIT+SMEM_SIZE_PER_TB_4BIT)        // used in fp6_linear.cu, Kernel_Ex().
/******************** Gloabl Memory Layout For QUANTIZED DATA *******************/
#define NUM_INT4_PER_WARP_1BIT     (WEIGHT_PER_WARP*BIT_WIDTH_1/128)                                          // 32
#define NUM_INT4_PER_WARP_2BIT     (WEIGHT_PER_WARP*BIT_WIDTH_2/128)                                          // 64
#define NUM_INT4_PER_WARP_4BIT     (WEIGHT_PER_WARP*BIT_WIDTH_4/128)                                          // 128

/*
 * C = A*B
 * A: row major with ahead-of-time layout transformation, FP6
 * B: col major, FP16
 * C: col major, FP16
 */
 template<typename TilingConfig, typename OutputDataType, int EXPONENT, int MANTISSA>
__global__ void QUANT_GEMM_Kernel(const uint4* Weight, const half* Scales,
                                  const half *B,
                                  OutputDataType* C,
                                  const size_t M_Global, const size_t N_Global, const size_t K_Global,
                                  int Split_K)
{
  #ifdef DEBUG_MODE
    assert(K_Global%TilingConfig::TILE_K==0);
    assert(M_Global%TilingConfig::TILE_M==0);
    assert( gridDim.y == Split_K * (M_Global/TilingConfig::TILE_M));
  #endif
  // 1+2+4 weight split
  constexpr int BIT_WIDTH = 1 + EXPONENT + MANTISSA;
  constexpr int USE_SEG_1BIT = BIT_WIDTH & 1;
  constexpr int USE_SEG_2BIT = BIT_WIDTH & 2;
  constexpr int USE_SEG_4BIT = BIT_WIDTH & 4;
  const uint4* Weight_1bit = Weight;
  const uint4* Weight_2bit = Weight_1bit + (USE_SEG_1BIT ? M_Global*K_Global*BIT_WIDTH_1/128 : 0);
  const uint4* Weight_4bit = Weight_2bit + (USE_SEG_2BIT ? M_Global*K_Global*BIT_WIDTH_2/128 : 0);
  // Dynamic shared memory for FP16 A tiles， 128 Bytes aligned
  extern __shared__ __align__(128) half smem[];
  half (*smem_array)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = reinterpret_cast<half (*)[WARP_K+PADDING_SHARED_MEM_FOR_B_8]> ( smem + SMEM_SIZE_PER_TB_A_TILE/2 ); // Dynamic shared memory for FP16 B tiles
  __shared__ half QuantScales[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // Thread Block Mapping, considering SplitK
  const size_t BatchID = blockIdx.y / (M_Global/TilingConfig::TILE_M);
  const size_t x = blockIdx.x;                                     // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t y = blockIdx.y % (M_Global/TilingConfig::TILE_M);   // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t Tile_Start_M = y * TilingConfig::TILE_M;
  const size_t Tile_Start_N = x * TilingConfig::TILE_N;
  const size_t NumColumnToCopy = (N_Global-Tile_Start_N) < TilingConfig::TILE_N ? (N_Global-Tile_Start_N) : TilingConfig::TILE_N;
  const size_t NumBlock_K = K_Global/TilingConfig::TILE_K;
  const size_t AverageNumBlock_K = NumBlock_K/Split_K;
  const size_t ExtraNumBlock_K   = NumBlock_K - AverageNumBlock_K * Split_K;
  size_t NumIter = AverageNumBlock_K;
  size_t StartBlockID_K = AverageNumBlock_K*BatchID;
  if(BatchID<ExtraNumBlock_K){
    NumIter ++;
    StartBlockID_K += BatchID;
  }
  else
    StartBlockID_K += ExtraNumBlock_K;
  // Warp ID.
  const int warpId = threadIdx.x / WARP_SIZE;
  int WARP_i = warpId / TilingConfig::BLOCK_COL_WARPS;  // WARP_i: row number;  WARP_j: column number
  //int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
  // Global Memory Address for Matrix A (Weight) /////////////////////////////////////////////////////////////////////////
  // StartPTR for each ThreadBlock(TB)
  const uint4* TB_StartGPTR_A_1BIT = Weight_1bit + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K * NUM_INT4_PER_WARP_1BIT;
  const uint4* TB_StartGPTR_A_2BIT = Weight_2bit + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K * NUM_INT4_PER_WARP_2BIT;
  const uint4* TB_StartGPTR_A_4BIT = Weight_4bit + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K * NUM_INT4_PER_WARP_4BIT;
  // StartPTR for each WARP.
  const uint4* WARP_StartGPTR_A_1BIT  = TB_StartGPTR_A_1BIT + WARP_i * NumBlock_K * NUM_INT4_PER_WARP_1BIT;
  const uint4* WARP_StartGPTR_A_2BIT  = TB_StartGPTR_A_2BIT + WARP_i * NumBlock_K * NUM_INT4_PER_WARP_2BIT;
  const uint4* WARP_StartGPTR_A_4BIT  = TB_StartGPTR_A_4BIT + WARP_i * NumBlock_K * NUM_INT4_PER_WARP_4BIT;
  // StartPTR for each WARP, considering SplitK
  const size_t WARP_Start_UnitID_K = StartBlockID_K;
  WARP_StartGPTR_A_1BIT  += WARP_Start_UnitID_K * NUM_INT4_PER_WARP_1BIT;
  WARP_StartGPTR_A_2BIT  += WARP_Start_UnitID_K * NUM_INT4_PER_WARP_2BIT;
  WARP_StartGPTR_A_4BIT  += WARP_Start_UnitID_K * NUM_INT4_PER_WARP_4BIT;
  // Copying A tile from Global to Shared, using double-buffer //////////////////////////////////////////////////////////
  // StartSPTR for each ThreadBlock
  uint32_t* AFrag_1BIT_SPTR = reinterpret_cast<uint32_t*>(smem);
  uint32_t* AFrag_2BIT_SPTR = AFrag_1BIT_SPTR + SMEM_SIZE_PER_TB_1BIT/4;
  uint32_t* AFrag_4BIT_SPTR = AFrag_2BIT_SPTR + SMEM_SIZE_PER_TB_2BIT/4;   // 8 buffers including double buffers, 12 for trible buffers
  // StartSPTR for each WARP
  AFrag_1BIT_SPTR += warpId * SMEM_SIZE_PER_WARP_1BIT/4;
  AFrag_2BIT_SPTR += warpId * SMEM_SIZE_PER_WARP_2BIT/4;
  AFrag_4BIT_SPTR += warpId * SMEM_SIZE_PER_WARP_4BIT/4;
  // Pre-fetch of A tile
  for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
    if(USE_SEG_1BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_1BIT>(AFrag_1BIT_SPTR+i*SMEM_SIZE_PER_WARP_1BIT/4*4, WARP_StartGPTR_A_1BIT);
    if(USE_SEG_2BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_2BIT>(AFrag_2BIT_SPTR+i*SMEM_SIZE_PER_WARP_2BIT/4*4, WARP_StartGPTR_A_2BIT);
    if(USE_SEG_4BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_4BIT>(AFrag_4BIT_SPTR+i*SMEM_SIZE_PER_WARP_4BIT/4*4, WARP_StartGPTR_A_4BIT);
    WARP_StartGPTR_A_1BIT += SMEM_SIZE_PER_WARP_1BIT/16;
    WARP_StartGPTR_A_2BIT += SMEM_SIZE_PER_WARP_2BIT/16;
    WARP_StartGPTR_A_4BIT += SMEM_SIZE_PER_WARP_4BIT/16;
  }
  // Global Memory Address for Matrix A (QuantScale) /////////////////////////////////////////////////////////////////////
  const half* TB_StartGPTR_A_Scale    = Scales + (y*TilingConfig::BLOCK_ROW_WARPS) * 64;
  const half* WARP_StartGPTR_A_Scales = TB_StartGPTR_A_Scale + WARP_i * 64;
  CopyFromGlobalToShared_Scales(QuantScales+WARP_i*64, WARP_StartGPTR_A_Scales);
  // Copying B tile from Global to Shared, considering SplitK /////////////////////////////////////////////////////////////
  const half *BTile_GPTR = B + Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K;
  for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
    CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS> (smem_array+i*TilingConfig::TILE_N, BTile_GPTR, K_Global, NumColumnToCopy);
    BTile_GPTR += TilingConfig::TILE_K;
  }
  // Register Allocation for A,B, and C, Initilazed to Zeros /////////////////////////////////////////////////////////////////////
  constexpr int NumRegSets_a = WARP_ROW_MMA_TENSORS;                                                                  // 1 set = 4 registers, containing a 16*16 MMA block
  constexpr int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
  uint32_t a  [NumRegSets_a * PIPELINE_LEVEL_SMEM][4];      // double/Trible buffer is used // Registers to store decompressed FP6
  uint32_t b  [NumRegSets_b * PIPELINE_LEVEL_SMEM][4];      // double/Triple buffer is used // Register to store FP16 B matrix (a slice)
  float c[NumRegSets_a * NumRegSets_b][REG_PER_THREAD_C_TENSOR_16_16];
  for(int i=0; i<NumRegSets_a * NumRegSets_b; i++)
    for(int j=0; j<REG_PER_THREAD_C_TENSOR_16_16; j++)
      c[i][j] = 0.0f;
  //
  #if __CUDA_ARCH__ >= 800
  cp_async_wait_all();
  #endif
  __syncthreads();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  uint32_t Scales_RPTR[4]; // 4 Registers per thread for Quantization Scales
  ExtractFromSharedToReg_Scales(Scales_RPTR, QuantScales + WARP_i*64);
  // Initializing the Software Pipeline: writing registers. ////////////////////////////////////////////////////////////////////////////////////////////////
  initialize_mma_slice<TilingConfig, EXPONENT, MANTISSA>(a, b, AFrag_1BIT_SPTR, AFrag_2BIT_SPTR, AFrag_4BIT_SPTR, smem_array, Scales_RPTR);
  // The outer loop. /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #pragma unroll(1)
  for (size_t tile_id_k = 0; tile_id_k < NumIter; tile_id_k++)
  {
    // Trible-Buffer for A Tile
    uint32_t* __restrict__ read_SPTR_Frag_1bit  = AFrag_1BIT_SPTR + ((tile_id_k+0) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_1BIT/4*4; // 512  (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    uint32_t* __restrict__ read_SPTR_Frag_2bit  = AFrag_2BIT_SPTR + ((tile_id_k+0) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_2BIT/4*4; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    uint32_t* __restrict__ read_SPTR_Frag_4bit  = AFrag_4BIT_SPTR + ((tile_id_k+0) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_4BIT/4*4; // 2048 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    uint32_t* __restrict__ read2_SPTR_Frag_1bit = AFrag_1BIT_SPTR + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_1BIT/4*4;
    uint32_t* __restrict__ read2_SPTR_Frag_2bit = AFrag_2BIT_SPTR + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_2BIT/4*4;
    uint32_t* __restrict__ read2_SPTR_Frag_4bit = AFrag_4BIT_SPTR + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_4BIT/4*4;
    uint32_t* __restrict__ write_SPTR_Frag_1bit = AFrag_1BIT_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_1BIT/4*4; // 512  (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    uint32_t* __restrict__ write_SPTR_Frag_2bit = AFrag_2BIT_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_2BIT/4*4; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    uint32_t* __restrict__ write_SPTR_Frag_4bit = AFrag_4BIT_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_PER_WARP_4BIT/4*4; // 2048 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // Trible-Buffer for B Tile
    // MODIFICATION NOTE: to support MSVC, half __restrict__ (*read_SPTR ) is changed to below. similarly for read2_SPTR and write_SPTR.
    half (* __restrict__ read_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k+0)  % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
    half (* __restrict__ read2_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
    half (* __restrict__ write_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
    //
    bool GlobalCopy = (tile_id_k+PIPELINE_LEVEL_GMEM-1) < NumIter;
    // Copying A tile from Global to Register, Bypassing L1, using double-buffer
    if(USE_SEG_1BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_1BIT>(write_SPTR_Frag_1bit, WARP_StartGPTR_A_1BIT, GlobalCopy);
    if(USE_SEG_2BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_2BIT>(write_SPTR_Frag_2bit, WARP_StartGPTR_A_2BIT, GlobalCopy);
    if(USE_SEG_4BIT) CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP_4BIT>(write_SPTR_Frag_4bit, WARP_StartGPTR_A_4BIT, GlobalCopy);
    // copying B tile from GlobalMemory to SharedMemory
    CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS> (write_SPTR, BTile_GPTR, K_Global, NumColumnToCopy, GlobalCopy);
    #if __CUDA_ARCH__ >= 800
    cp_async_group_commit();
    #endif
    core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read_SPTR_Frag_1bit, read_SPTR_Frag_2bit, read_SPTR_Frag_4bit, read_SPTR, Scales_RPTR, 1); // read_SPTR_Frag_2bit, read_SPTR_Frag_4bit are different for each WARP; read_SPTR is shared among WARPs
    core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read_SPTR_Frag_1bit, read_SPTR_Frag_2bit, read_SPTR_Frag_4bit, read_SPTR, Scales_RPTR, 2);
    core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read_SPTR_Frag_1bit, read_SPTR_Frag_2bit, read_SPTR_Frag_4bit, read_SPTR, Scales_RPTR, 3);
    // Barriers and Synchronizations
    #if __CUDA_ARCH__ >= 800
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    #endif
    __syncthreads();
    core_mma_slice<TilingConfig, EXPONENT, MANTISSA>(c, a, b, read2_SPTR_Frag_1bit, read2_SPTR_Frag_2bit, read2_SPTR_Frag_4bit, read2_SPTR, Scales_RPTR, 0);
    // Updating global PTRs
    WARP_StartGPTR_A_1BIT += SMEM_SIZE_PER_WARP_1BIT/16;  // 2KB/16=128 (1)/16: int4*+1 = char*+16
    WARP_StartGPTR_A_2BIT += SMEM_SIZE_PER_WARP_2BIT/16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
    WARP_StartGPTR_A_4BIT += SMEM_SIZE_PER_WARP_4BIT/16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
    BTile_GPTR += TilingConfig::TILE_K;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store the C fragments to shared memory.
  float (*smem_CFrag) [TilingConfig::TILE_M+PADDING_SHARED_MEM_FOR_C_4] =
        reinterpret_cast <float (*)[TilingConfig::TILE_M+PADDING_SHARED_MEM_FOR_C_4]> (smem);
  StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
  __syncthreads();
  // Now that shared memory contains all the D tiles, stream them to global memory.
  OutputDataType* BlockGlobalPTR = C + BatchID*(M_Global*N_Global) + Tile_Start_M + Tile_Start_N*M_Global;
  for(size_t i=warpId; i<NumColumnToCopy; i+=TilingConfig::BLOCK_WARPS)    // i-th column
    #pragma unroll
    for(size_t j=threadIdx.x%WARP_SIZE; j<TilingConfig::TILE_M; j+=WARP_SIZE) // j-th row
    {
      if constexpr (std::is_same<OutputDataType, half>::value)   BlockGlobalPTR[j+i*M_Global] = __float2half_rn(smem_CFrag[i][j]);
      else                                            BlockGlobalPTR[j+i*M_Global] = smem_CFrag[i][j];
    }
}
