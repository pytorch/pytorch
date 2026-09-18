#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/IGammaShapeDerivative.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {
namespace {

void igamma_grad_a_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "igamma_grad_a_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t a, scalar_t x) -> scalar_t {
          return static_cast<scalar_t>(-igamma_grad_detail::derivative_q(
              static_cast<double>(a), static_cast<double>(x)));
        });
      });
}

} // namespace

REGISTER_DISPATCH(igamma_grad_a_stub, &igamma_grad_a_kernel)

} // namespace at::native
