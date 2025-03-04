#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/ck_gemm.h>
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

template <>
void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  dispatch_half_gemm(CUDABLAS_GEMM_ARGS(at::Half));
}

} // namespace at::native
