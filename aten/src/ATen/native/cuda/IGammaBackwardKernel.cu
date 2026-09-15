#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/IGammaShapeDerivative.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

void igamma_grad_a_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_grad_a_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t x) -> scalar_t {
      return static_cast<scalar_t>(-igamma_grad_detail::derivative_q(
          static_cast<double>(a), static_cast<double>(x)));
    });
  });
}

REGISTER_DISPATCH(igamma_grad_a_stub, &igamma_grad_a_kernel_cuda)

} // namespace at::native
