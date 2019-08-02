#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/PointwiseOps.h>
#include <THC/THCNumerics.cuh>

namespace at { namespace native {

static void addcdiv_cuda_kernel(TensorIterator& iter, Scalar value) {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "addcdiv_cuda", [&]() {
      auto alpha = value.to<scalar_t>();
      gpu_kernel(iter, [alpha]GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        return THCNumerics<scalar_t>::add(a, THCNumerics<scalar_t>::mul(alpha,THCNumerics<scalar_t>::div(b, c)));
      });
    });
  }
}

REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cuda_kernel);

}} // namespace at::native
