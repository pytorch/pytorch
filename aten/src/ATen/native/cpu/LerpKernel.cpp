#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {
namespace {

static void lerp_kernel(
    Tensor& ret,
    const Tensor& self,
    const Tensor& end,
    Scalar weight) {
  auto builder = at::TensorIterator::Builder();
  builder.add_output(ret);
  builder.add_input(self);
  builder.add_input(end);
  auto iter = builder.build();

  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "lerp_kernel", [&] {
    scalar_t weight_val = weight.to<scalar_t>();
    at::native::binary_kernel(*iter, [=](scalar_t self_val, scalar_t end_val) {
      return (weight_val < 0.5)
          ? self_val + weight_val * (end_val - self_val)
          : end_val - (end_val - self_val) * (1 - weight_val);
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(lerp_stub, &lerp_kernel);

} // namespace native
} // namespace at
