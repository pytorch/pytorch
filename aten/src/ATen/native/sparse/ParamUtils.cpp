#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

std::tuple<Tensor, Tensor, int64_t> softmax_sparse_input_preprocessing(
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
  Tensor output = at::native::empty_like_sparse_coo(input);
  int64_t dim = c10::maybe_wrap_dim(dim_, input.dim());
  return std::make_tuple(input, output, dim);
}

std::tuple<Tensor, Tensor, Tensor, int64_t> softmax_backward_sparse_input_preprocessing(
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

  Tensor grad_input = at::native::empty_like_sparse_coo(output);
  TORCH_CHECK(
      grad.sparse_dim() == output.sparse_dim(),
      ": grad and output sparse dimensions must be equal");
  return std::make_tuple(grad_input, grad, output, dim);
}

} // namespace at::native
