#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

static void lerp_kernel_scalar(
    Tensor& ret,
    const Tensor& self,
    const Tensor& end,
    Scalar weight) {
  // lerp() only uses TensorIterator for CPU. Since TensorIterator would
  // would attempt to promote types inconsistent with the CUDA implementation,
  // restrict types explicitly here
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(), " for `end` but got dtype ", end.dtype());
  auto iter = TensorIterator::binary_op(ret, self, end,
                                        /*check_mem_overlap=*/true);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(ret.scalar_type(), "lerp_kernel_scalar", [&] {
    using value_t = typename ztype<scalar_t>::value_t;
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

static void lerp_kernel_tensor(
    Tensor& ret,
    const Tensor& self,
    const Tensor& end,
    const Tensor& weights) {
  // lerp() only uses TensorIterator for CPU. Since TensorIterator would
  // would attempt to promote types inconsistent with the CUDA implementation,
  // restrict types explicitly here
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(), " for `end` but got dtype ", end.dtype());
  TORCH_CHECK(self.dtype() == weights.dtype(), "expected dtype ", self.dtype(), " for `weights` but got dtype ", end.dtype());
  auto iter = TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(ret);
  iter.add_input(self);
  iter.add_input(end);
  iter.add_input(weights);
  iter.build();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(ret.scalar_type(), "lerp_kernel_tensor", [&] {
    using value_t = typename ztype<scalar_t>::value_t;
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

REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_kernel_scalar);
REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_kernel_tensor);

} // namespace native
} // namespace at
