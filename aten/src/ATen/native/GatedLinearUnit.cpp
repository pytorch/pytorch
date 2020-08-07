#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>

#include <utility>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(glu_stub);
DEFINE_DISPATCH(glu_backward_stub);

namespace {
// Tuple, containing <new_shape, new_dimension>
// The `new_dimension` is a the dimension of split, but with correction for wrap
// The new shape has the "half size" across the `new_dimension`
using SizesAndDim = std::pair<std::vector<int64_t>, int64_t>;
using TensorPair = std::pair<Tensor, Tensor>;

SizesAndDim get_sizes_and_wrap_dim(const Tensor& self, int64_t dim) {
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
  return std::make_pair(newSizes, wrap_dim);
}

TensorPair make_split_pair(const Tensor& self, int64_t half_size, int64_t dim) {
  Tensor firstHalf = self.narrow(dim, 0, half_size);
  Tensor secondHalf = self.narrow(dim, half_size, half_size);
  return std::make_pair(firstHalf, secondHalf);
}
}  // namespace

Tensor& glu_out(Tensor &result, const Tensor& self, int64_t dim) {
  SizesAndDim size_dim = get_sizes_and_wrap_dim(self, dim);
  result.resize_(size_dim.first);
  auto half_size = size_dim.first[size_dim.second];
  TensorPair split_self = make_split_pair(self, half_size, size_dim.second);

  auto iter = TensorIterator::binary_op(result, split_self.first,
                                        split_self.second);
  glu_stub(iter.device_type(), iter);
  return result;
}

Tensor glu(const Tensor& self, int64_t dim) {
  auto result = at::empty({0}, self.options());
  return at::glu_out(result, self, dim);
}

Tensor& glu_backward_out(Tensor& grad_input,
    const Tensor& grad_output, const Tensor& input, int64_t dim) {
  SizesAndDim size_dim = get_sizes_and_wrap_dim(input, dim);
  grad_input.resize_as_(input);
  auto half_size = size_dim.first[size_dim.second];
  TensorPair split_input = make_split_pair(input, half_size, size_dim.second);
  TensorPair split_grad_input = make_split_pair(grad_input, half_size,
                                                size_dim.second);

  at::sigmoid_out(split_grad_input.first, split_input.second);
  // for second gradinput half, can get a better performance by fusion
  auto iter = at::TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .add_output(split_grad_input.second)
    .add_input(split_grad_input.first)
    .add_input(split_input.first)
    .add_input(grad_output)
    .build();
  glu_backward_stub(iter.device_type(), iter);
  split_grad_input.first.mul_(grad_output);
  return grad_input;
}

Tensor glu_backward(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return at::glu_backward_out(grad_input, grad_output, input, dim);
}

} // at::native
} // at
