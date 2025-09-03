#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm.h>
#if defined(USE_ROCM_CK_GEMM)
#include <ATen/native/hip/ck_gemm_template.h>

#include <ck/utility/sequence.hpp>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

namespace at::native {

void dispatch_half_gemm(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
#if 0
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.

  bool transa_ = std::tolower(transa) != 'n';
  bool transb_ = std::tolower(transb) != 'n';

  if (use_padding) {
    if (m <= 128) {
      if(transa_ && transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(transa_ && !transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && !transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }



    } else {

      if(transa_ && transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(transa_ && !transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            false,
            true>(CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && !transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
    }
  } else {
    {
      if(transa_ && transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            1,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(transa_ && !transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            1,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>
            1,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && !transb_) {
          gemm_impl<
            at::Half,
            256,
            256,
            128,
            32,
            4,
            4,
            32,
            32,
            4,
            2,
            S<8,32,1>,
            S<1,0,2>,
            S<1,0,2>,
            1,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
    }
  }
#endif
}
void dispatch_half_gemm_wmma(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.

  bool transa_ = std::tolower(transa) != 'n';
  bool transb_ = std::tolower(transb) != 'n';

  if (use_padding) {
      if(transa_ && transb_) { // col , col
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(transa_ && !transb_) { // row, col
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && transb_) { //col, row
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && !transb_) { //row, row
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            true,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
  } else {
         if(transa_ && transb_) { // col , col
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            true,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(transa_ && !transb_) { // row, col
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            true,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && transb_) { //col, row
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>,
            8,
            false,
            false,
            true>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else if(!transa_ && !transb_) { //row, row
          gemm_impl_wmma<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1,  8>, 8,
            false,
            false,
            false>
            (CUDABLAS_GEMM_ARGS(at::Half));
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
  }
}

template <>
void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  if (at::detail::getCUDAHooks().isGPUArch({"gfx1100"})) {
    dispatch_half_gemm_wmma(CUDABLAS_GEMM_ARGS(at::Half));
  } else{
    dispatch_half_gemm(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

} // namespace at::native
#endif // USE_ROCM_CK_GEMM
