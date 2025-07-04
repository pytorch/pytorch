#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm/ck_gemm_template.h>

namespace at::native {
    // TODO: need to decide padding and layout here
void
ck_gemm_kernel_bf16bf16bf16_256_128x128x64_16x16_4x4_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3(
        CUDABLAS_GEMM_ARGTYPES(at::BFloat16),
        bool use_padding) {
  // get padding and layout
  bool transa_ = std::tolower(transa) != 'n';
  bool transb_ = std::tolower(transb) != 'n';
  // create outer layout loop
  if (use_padding) {
    if (transa_ & transb_) {            // TT
      gemm_impl<
         at::BFloat16,
          256,            // BLOCK_SIZE
          128,            // MBLOCK
          128,            // NBLOCK
          64,             // KBLOCK
          8,              // AK1
          8,              // BK1
          16,             // MPER_XDL
          16,             // NPER_XDL
          4,              // MPER_WAVE
          4,              // NPER_WAVE
          S<8,32,1>,      // ABLOCK_CLUSTER_LENS
          S<0,2,1>,       // ABLOCK_CLUSTER_ORDER
          S<0,2,1>,       // ABLOCK_SRC_ORDER
          1,              // ABLOCK_VECTOR_DIM
          4,              // ABLOCK_SCALAR_VEC
          8,              // ABLOCK_SCALAR_VEC_AK1
          0,              // ABLOCK_LDS_EXTRAM
          S<8,32,1>,      // BBLOCK_CLUSTER_LENS
          S<1,0,2>,       // BBLOCK_CLUSTER_ORDER
          S<1,0,2>,       // BBLOCK_SRC_ORDER
          2,              // BBLOCK_VECTOR_DIM
          8,              // BBLOCK_SCALAR_VEC
          8,              // BBLOCK_SCALAR_VEC_AK1
          0,              // BBLOCK_LDS_EXTRAN
          1,              // CMPER_WAVE
          1,              // CNPER_WAVE
          S<1, 32, 1, 8>, // BLOCK_CLUSTER_LENS
          S<4>,           // CDE_SCALAR_VEC
          true,           // PADDING
          true,           // TRANSA
          true>           // TRANSB
          (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else if (transa_ && !transb_) {   // TN
      gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            16,
            16,
            4,
            4,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else if (!transa_ && transb_) {   // NT
        gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            16,
            16,
            4,
            4,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else if (!transa_ && !transb_) {  // NN
        gemm_impl<
           at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            16,
            16,
            4,
            4,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else {
      // For sanity
      TORCH_CHECK(false, "unreachable:ck_gemm_kernel_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8")
    } // End layout conditional

  } else { // !use_padding
    if (transa_ & transb_) {            // TT
      gemm_impl<
         at::BFloat16,
          256,            // BLOCK_SIZE
          128,            // MBLOCK
          128,            // NBLOCK
          64,             // KBLOCK
          8,              // AK1
          8,              // BK1
          16,             // MPER_XDL
          16,             // NPER_XDL
          4,              // MPER_WAVE
          4,              // NPER_WAVE
          S<8,32,1>,      // ABLOCK_CLUSTER_LENS
          S<0,2,1>,       // ABLOCK_CLUSTER_ORDER
          S<0,2,1>,       // ABLOCK_SRC_ORDER
          1,              // ABLOCK_VECTOR_DIM
          4,              // ABLOCK_SCALAR_VEC
          8,              // ABLOCK_SCALAR_VEC_AK1
          0,              // ABLOCK_LDS_EXTRAM
          S<8,32,1>,      // BBLOCK_CLUSTER_LENS
          S<1,0,2>,       // BBLOCK_CLUSTER_ORDER
          S<1,0,2>,       // BBLOCK_SRC_ORDER
          2,              // BBLOCK_VECTOR_DIM
          8,              // BBLOCK_SCALAR_VEC
          8,              // BBLOCK_SCALAR_VEC_AK1
          0,              // BBLOCK_LDS_EXTRAN
          1,              // CMPER_WAVE
          1,              // CNPER_WAVE
          S<1, 32, 1, 8>, // BLOCK_CLUSTER_LENS
          S<4>,           // CDE_SCALAR_VEC
          false,          // PADDING
          true,           // TRANSA
          true>           // TRANSB
          (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else if (transa_ && !transb_) {   // TN
      gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            16,
            16,
            4,
            4,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else if (!transa_ && transb_) {   // NT
        gemm_impl<
            at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            16,
            16,
            4,
            4,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else if (!transa_ && !transb_) {  // NN
        gemm_impl<
           at::BFloat16,
            256,
            128,
            128,
            64,
            8,
            8,
            16,
            16,
            4,
            4,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            2,
            8,
            8,
            0,
            S<8,32,1>,
            S<0,2,1>,
            S<0,2,1>,
            1,
            4,
            8,
            0,
            1,
            1,
            S<1, 32, 1, 8>,
            S<4>,
            false,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else {
      // For sanity
      TORCH_CHECK(false, "unreachable: ck_gemm_kernel_256_128x128x64_16x16_4x4_8x32x1_8x32x1_1x32x1x8")
    } // End layout conditional

  }
  // call kernel
}
}; // namespace at::native
