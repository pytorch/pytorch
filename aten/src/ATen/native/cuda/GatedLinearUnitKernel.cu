#include <ATen/native/GatedLinearUnit.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>

namespace at { namespace native {

void glu_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "glu_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      using accscalar_t = acc_type<scalar_t, true>;
      const accscalar_t sigNum = accscalar_t(1) / (accscalar_t(1) + std::exp(accscalar_t(-b)));
      return a * sigNum;
    });
  });
}

REGISTER_DISPATCH(glu_stub, &glu_kernel_cuda);
}}
