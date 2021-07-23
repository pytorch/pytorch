#include <ATen/native/UnfoldBackward.h>

namespace at { namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(unfold_backward_stub);

Tensor unfold_backward(
  const Tensor& grad,
  IntArrayRef input_sizes,
  int64_t dim,
  int64_t size,
  int64_t step
) {
  auto grad_input = at::zeros(input_sizes, grad.options());

  unfold_backward_stub(
    grad.device().type(),
    grad_input,
    grad,
    dim, size, step
  );

  return grad_input;
}

}} // namespace at::native
