#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

Tensor sparse_hardshrink(const Tensor& input_, Scalar lambd) {
  TORCH_INTERNAL_ASSERT(input_.is_sparse());
  auto input = input_.coalesce();
  Tensor result = at::native::empty_like(input);
  if (input.numel() == 0) {
    return result;
  }
  auto indices = input._indices().contiguous();
  auto input_values = input._values().contiguous();

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

  result_indices.resize_as_(indices);
  result_indices.copy_(indices);

  result_values.resize_as_(input_values);

  result_values.copy_(at::native::hardshrink(input_values, lambd));

  return result;
}

Tensor sparse_hardshrink_backward(
    const Tensor& grad,
    const Tensor& input_,
    Scalar lambd) {
  TORCH_INTERNAL_ASSERT(input_.is_sparse());
  auto input = input_.coalesce();
  Tensor result = at::native::empty_like(input);
  if (input.numel() == 0) {
    return result;
  }

  auto indices = input._indices().contiguous();
  auto input_values = input._values().contiguous();

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

  result_indices.resize_as_(indices);
  result_indices.copy_(indices);

  result_values.resize_as_(input_values);

  auto grad_values = grad._values().contiguous();
  result_values.copy_(
      at::native::hardshrink_backward(grad_values, input_values, lambd));
  return result;
}

} // namespace native
} // namespace at
