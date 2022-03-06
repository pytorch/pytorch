#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/DispatchStub.h>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

// TODO: review jiterating igamma and igammac if/when a persistent (across processes)
//   cache is implemented, because they take a VERY long time to compile
// TODO: it's also odd these ops use gpu_kernel_with_scalars

// end of regularized lower & upper incomplete gamma

namespace at { namespace native {
namespace{

const char igamma_name[] = "igamma";
void igamma_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars</*name=*/igamma_name,
                                     /*return_dtype=*/ scalar_t,
                                     /*f_inputs_dtype=*/ scalar_t>(iter, igamma_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t q) -> scalar_t {
        return calc_igamma<scalar_t, /*is_cuda=*/true>(x, q);
      });
    });
  #endif //jiterator
}

const char igammac_name[] = "igammac";
void igammac_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars</*name=*/igammac_name,
                                     /*return_dtype=*/ scalar_t,
                                     /*f_inputs_dtype=*/ scalar_t>(iter, igammac_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t q) -> scalar_t {
        return calc_igammac<scalar_t, /*is_cuda=*/true>(x, q);
      });
    });
  #endif //jiterator
}

const char gammaincinv_name[] = "calc_gammaincinv";
void gammaincinv_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "gammaincinv_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars</*name=*/gammaincinv_name,
                                     /*return_dtype=*/ scalar_t,
                                     /*f_inputs_dtype=*/ scalar_t>(iter, gammaincinv_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "gammaincinv_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t q) -> scalar_t {
        return calc_gammaincinv<scalar_t, /*is_cuda=*/true>(x, q);
      });
    });
  #endif //jiterator
}

const char gammainccinv_name[] = "calc_gammainccinv";
void gammainccinv_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "gammainccinv_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars</*name=*/gammainccinv_name,
                                     /*return_dtype=*/ scalar_t,
                                     /*f_inputs_dtype=*/ scalar_t>(iter, gammainccinv_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "gammainccinv_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t q) -> scalar_t {
        return calc_gammainccinv<scalar_t, /*is_cuda=*/true>(x, q);
      });
    });
  #endif //jiterator
}

}

REGISTER_DISPATCH(igamma_stub, &igamma_kernel_cuda);
REGISTER_DISPATCH(igammac_stub, &igammac_kernel_cuda);
REGISTER_DISPATCH(gammaincinv_stub, &gammaincinv_kernel_cuda);
REGISTER_DISPATCH(gammainccinv_stub, &gammainccinv_kernel_cuda);

// DO NOT ADD ANY NEW KERNELS HERE
// CUDA compilation times grow quickly.  It's perfectly acceptable to have a file per kernel.

}} // namespace at::native
