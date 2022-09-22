#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

void lerp_scalar_kernel(at::TensorIteratorBase& iter, const Scalar& weight) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_scalar", [&] {
    using value_t = typename c10::scalar_value_type<scalar_t>::type;
    scalar_t weight_val = weight.to<scalar_t>();
    at::native::cpu_kernel(
        iter,
        [weight_val](scalar_t self_val, scalar_t end_val) {
          return (zabs<scalar_t, value_t>(weight_val) < 0.5)
              ? self_val + weight_val * (end_val - self_val)
              : end_val - (end_val - self_val) * (scalar_t(1) - weight_val);
        });
  });
}

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_tensor", [&] {
    using value_t = typename c10::scalar_value_type<scalar_t>::type;
    at::native::cpu_kernel(
        iter,
        [](scalar_t self_val, scalar_t end_val, scalar_t weight_val) {
          return (zabs<scalar_t, value_t>(weight_val) < 0.5)
              ? self_val + weight_val * (end_val - self_val)
              : end_val - (end_val - self_val) * (scalar_t(1) - weight_val);
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel);
REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel);

} // namespace native
} // namespace at
