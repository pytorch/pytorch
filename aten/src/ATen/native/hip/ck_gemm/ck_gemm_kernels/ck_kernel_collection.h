#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>

namespace at::native {
/* Naming convention:
 * ck_gemm_kernel_AdtypeBdtypeOUTdtyp_BLOCKSIZE_MBLOCKxNBLOCKxKBLOCK_MPERXDL_NPERXDL_MPERWAVExNPERWAVE_<ABLOCK_CLUSTER_LENS>_<BBLOCK_CLUSTER_LENS>_<BLOCK_CLUSTER_LENS>_CDE_SCALAR_VEC_Intrawave_v3
 */

// Small
void
ck_gemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
        CUDABLAS_GEMM_ARGTYPES(at::BFloat16),
        bool use_padding);

// Medium
void
ck_gemm_kernel_bf16bf16bf16_256_128x128x64_16x16_4x4_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
        CUDABLAS_GEMM_ARGTYPES(at::BFloat16),
        bool use_padding);

// Large
void
ck_gemm_kernel_bf16bf16bf16_256_256x128x64_16x16_8x4_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
        CUDABLAS_GEMM_ARGTYPES(at::BFloat16),
        bool use_padding);

}; // namespace at::native
