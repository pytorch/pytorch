#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(glu_stub);
DEFINE_DISPATCH(glu_backward_stub);

Tensor& glu_out(const Tensor& self, int64_t dim, Tensor &result) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);
  // size output to half of input
  const int64_t selfSize = nIn / 2;
  auto newSizes = self.sizes().vec();
  newSizes[wrap_dim] = selfSize;
  result.resize_(newSizes);
  // half tensor
  Tensor firstHalf = self.narrow(wrap_dim, 0, selfSize);
  Tensor secondHalf = self.narrow(wrap_dim, selfSize, selfSize);

  auto iter = TensorIterator::binary_op(result, firstHalf, secondHalf);
  glu_stub(iter.device_type(), iter);
  return result;
}

Tensor glu(const Tensor& self, int64_t dim) {
  auto result = at::empty({0}, self.options());
  return at::glu_out(result, self, dim);
}

Tensor& glu_backward_out(const Tensor& grad_output, const Tensor& input, int64_t dim, Tensor& grad_input) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const int64_t nIn = input.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  grad_input.resize_as_(input);
  const int64_t inputSize = nIn / 2;
  // half tensor
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  Tensor gradInputfirstHalf = grad_input.narrow(wrap_dim, 0, inputSize);
  Tensor gradInputsecondHalf = grad_input.narrow(wrap_dim, inputSize, inputSize);

  at::sigmoid_out(gradInputfirstHalf, secondHalf);
  // for second gradinput half, can get a better performance by fusion
  auto iter = at::TensorIteratorConfig()
    .add_output(gradInputsecondHalf)
    .add_input(gradInputfirstHalf)
    .add_input(firstHalf)
    .add_input(grad_output)
    .build();
  glu_backward_stub(iter.device_type(), iter);
  gradInputfirstHalf.mul_(grad_output);
  return grad_input;
}

Tensor glu_backward(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return at::glu_backward_out(grad_input, grad_output, input, dim);
}

} // at::native
} // at
