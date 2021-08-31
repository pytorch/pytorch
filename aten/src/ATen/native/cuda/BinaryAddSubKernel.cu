#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <c10/cuda/CUDAGuard.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template <typename T>
struct AddFunctor {
  AddFunctor(T alpha) : alpha_(alpha) {}
  T alpha_;
  __device__ __forceinline__ T operator()(T a, T b) const __ubsan_ignore_undefined__ {
    return a + b * alpha_;
  }
};

void add_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, AddFunctor<opmath_t>(alpha_scalar.to<opmath_t>()));
  });
}

static void sub_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  add_kernel_cuda(iter, -alpha_scalar);
}

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);

}} // namespace at::native
