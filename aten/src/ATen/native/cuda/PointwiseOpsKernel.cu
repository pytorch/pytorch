#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/PointwiseOps.h>

namespace at { namespace native {

void addcdiv_cuda_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "addcdiv_cuda", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    gpu_kernel(iter, [alpha]GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return THCNumerics<T>::add(a, THCNumerics<T>::mul(alpha,THCNumerics<T>::div(b, c)));
    });
  }
}

REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cuda_kernel);

}} // namespace at::native
