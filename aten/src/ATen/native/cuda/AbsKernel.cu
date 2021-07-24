#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

template<typename scalar_t>
struct AbsFunctor {
  __device__ __forceinline__ scalar_t operator() (const scalar_t a) const {
    return std::abs(a);
  }
};

void abs_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, iter.dtype(), "abs_cuda", [&]() {
    gpu_kernel(iter, AbsFunctor<scalar_t>());
  });
}

REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda);

}} // namespace at::native
