#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

void abs_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, iter.dtype(), "abs_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::abs(a);
    });
  });
}

REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda);

}} // namespace at::native
