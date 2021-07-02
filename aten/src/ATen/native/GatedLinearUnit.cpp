#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(glu_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(glu_backward_stub);
DEFINE_DISPATCH(glu_backward_cuda_stub);

REGISTER_NO_CPU_DISPATCH(glu_backward_cuda_stub, glu_backward_cuda_fn);

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

  auto iter = TensorIterator::borrowing_binary_op(result, firstHalf, secondHalf);
  glu_stub(iter.device_type(), iter);
  return result;
}

Tensor glu(const Tensor& self, int64_t dim) {
  auto result = at::empty({0}, self.options());
  return at::glu_out(result, self, dim);
}

Tensor& glu_backward_cpu_out(const Tensor& grad_output, const Tensor& input,
                             int64_t dim, Tensor& grad_input) {
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

Tensor glu_backward_cpu(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return glu_backward_cpu_out(grad_output, input, dim, grad_input);
}

Tensor& glu_backward_cuda_out(const Tensor& grad_output, const Tensor& input,
                              int64_t dim, Tensor& grad_input) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  auto input_sizes = input.sizes();
  const int64_t nIn = input_sizes[wrap_dim];
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  resize_output(grad_input, input_sizes);

  DimVector iter_shape(input_sizes);
  iter_shape[wrap_dim] = nIn / 2;
  TORCH_CHECK(grad_output.sizes() == IntArrayRef{iter_shape});

  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_input(input)
    .add_input(grad_output)
    .resize_outputs(false)
    .declare_static_shape(iter_shape)
    .build();
  glu_backward_cuda_stub(iter.device_type(), iter, wrap_dim);
  return grad_input;
}

Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return glu_backward_cuda_out(grad_output, input, dim, grad_input);
}

} // at::native
} // at
