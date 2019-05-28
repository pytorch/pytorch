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
  auto builder = at::TensorIterator::Builder();
  builder.add_output(ret);
  builder.add_input(self);
  builder.add_input(end);
  auto iter = builder.build();

  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "lerp_kernel_scalar", [&] {
    scalar_t weight_val = weight.to<scalar_t>();
    auto weight_vec = Vec256<scalar_t>(weight_val);
    auto reversed_weight_vec = Vec256<scalar_t>(1 - weight_val);
    at::native::binary_kernel_vec(
        *iter,
        [=](scalar_t self_val, scalar_t end_val) {
          return (weight_val < 0.5)
              ? self_val + weight_val * (end_val - self_val)
              : end_val - (end_val - self_val) * (1 - weight_val);
        },
        [=](Vec256<scalar_t> start_vec, Vec256<scalar_t> end_vec) {
          return reversed_weight_vec * start_vec + weight_vec * end_vec;
        });
  });
}

static void lerp_kernel_tensor(
    Tensor& ret,
    const Tensor& self,
    const Tensor& end,
    const Tensor& weights) {
  auto builder = at::TensorIterator::Builder();
  builder.add_output(ret);
  builder.add_input(self);
  builder.add_input(end);
  builder.add_input(weights);
  auto iter = builder.build();
  AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "lerp_kernel_tensor", [&] {
    at::native::ternary_kernel(
        *iter,
        [=](scalar_t self_val, scalar_t end_val, scalar_t weight_val) {
          return (weight_val < 0.5)
              ? self_val + weight_val * (end_val - self_val)
              : end_val - (end_val - self_val) * (1 - weight_val);
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_kernel_scalar);
REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_kernel_tensor);

} // namespace native
} // namespace at
