#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UnfoldBackward.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/unfold_backward_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

DEFINE_DISPATCH(unfold_backward_stub);

Tensor unfold_backward(
  const Tensor& grad,
  IntArrayRef input_sizes,
  int64_t dim,
  int64_t size,
  int64_t step
) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  if (step >= size) {
    auto gI_unfolded = grad_input.unfold(dim, size, step);
    gI_unfolded.copy_(grad);
    return grad_input;
  }

  unfold_backward_stub(
    grad.device().type(),
    grad_input,
    grad,
    dim, size, step
  );

  return grad_input;
}

} // namespace at::native
