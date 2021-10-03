#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/ATen.h>
#include <tuple>

namespace at {
namespace native {

std::pair<Tensor, Tensor> softmax_sparse_input_preprocessing(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float,
    CheckedFrom function_name) {
  TORCH_INTERNAL_ASSERT(input_.is_sparse());
  TORCH_CHECK(
      !half_to_float,
      std::string(function_name) +
          ": with half to float conversion is not supported on " +
          input_.device().str());
  auto input = input_.coalesce();
  Tensor output = at::native::empty_like(input);
  TORCH_CHECK(
      dim_ >= 0 && dim_ < input.dim(),
      ": dim must be non-negative and less than input dimensions");
  return std::make_pair(input, output);
}

std::tuple<Tensor, Tensor, Tensor> softmax_backward_sparse_input_preprocessing(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_,
    CheckedFrom function_name) {
  TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
  checkSameSize(function_name, grad_arg, output_arg);

  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());

  auto grad = grad_.coalesce();
  auto output = output_.coalesce();

  Tensor grad_input = at::native::empty_like(output);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      ": dim must be non-negative and less than input dimensions");
  TORCH_CHECK(
      grad.sparse_dim() == output.sparse_dim(),
      ": grad and output sparse dimensions must be equal");
  return std::make_tuple(grad_input, grad, output);
}

} // namespace native
} // namespace at
