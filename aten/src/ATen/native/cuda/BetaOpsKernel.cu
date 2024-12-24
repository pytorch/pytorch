#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BetaOps.h>
#include <ATen/native/Math.h>

namespace at::native {

void betainc_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "betainc_cuda", [&]() {
    gpu_kernel_with_scalars_ternary(iter, []GPU_LAMBDA(scalar_t x, scalar_t a, scalar_t b) -> scalar_t {
        return calc_betainc(x, a, b);
    });
  });
}

REGISTER_DISPATCH(betainc_stub, &betainc_kernel_cuda)

} // namespace at::native
