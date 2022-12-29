#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/cpu/SampledReduceKernel.h>
#include <c10/util/Optional.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {

Tensor sampled_reduce_cpu(
    const Tensor& self,
    const Tensor& other,
    c10::string_view reduce,
    const c10::optional<Tensor>& left_index_opt,
    const c10::optional<Tensor>& right_index_opt) {

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> left_index_maybe_owned = at::borrow_from_optional_tensor(left_index_opt);
  const Tensor& left_index = *left_index_maybe_owned;
  const Tensor& right_index = c10::value_or_else(right_index_opt, [] {return Tensor();});

  TORCH_CHECK(self.dim() == 2,
      "sampled_reduce: Expected self to be 2D tensor, got ", self.dim(), "D tensor.");
  TORCH_CHECK(other.dim() == 2,
      "sampled_reduce: Expected other to be 2D tensor, got ", other.dim(), "D tensor.");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(),
      "sampled_reduce: Expected self and other have the same dtype.");
  TORCH_CHECK(self.dim() == other.dim(),
      "sampled_reduce: Expected self and other have the same size of dim 1.");
  TORCH_CHECK(self.numel() != 0 && other.numel() != 0,
      "sampled_reduce: self or other can't be empty.");

  bool left_index_has_value = left_index.defined();
  bool right_index_has_value = right_index.defined();

  if (left_index_has_value) {
    TORCH_CHECK(left_index.dim() == 1,
        "sampled_reduce: Expected left_index to be 1D tensor, got ", left_index.dim(), "D tensor.");
    TORCH_CHECK(left_index.scalar_type() == kLong || left_index.scalar_type() == kInt,
        "sampled_reduce: Expected left_index to be Int or Long tensor, got ", left_index.scalar_type());
    TORCH_CHECK(left_index.is_contiguous(),
        "sampled_reduce: Expected left_index to be contiguous.");
  }
  if (right_index_has_value) {
    TORCH_CHECK(right_index.dim() == 1,
        "sampled_reduce: Expected right_index to be 1D tensor, got ", right_index.dim(), "D tensor.");
    TORCH_CHECK(right_index.scalar_type() == kLong || right_index.scalar_type() == kInt,
        "sampled_reduce: Expected right_index to be Int or Long tensor, got ", right_index.scalar_type());
    TORCH_CHECK(right_index.is_contiguous(),
        "sampled_reduce: Expected right_index to be contiguous.");
  }

  int64_t output_height = 0;
  // check if the indices size match
  if (left_index_has_value && right_index_has_value) {
    TORCH_CHECK(left_index.scalar_type() == right_index.scalar_type(),
        "sampled_reduce: Expected left and right index to have the same dtype.");
    TORCH_CHECK(left_index.size(0) == right_index.size(0),
        "sampled_reduce: Expected left and right index to have the same size, got ",
        left_index.size(0), " and ", right_index.size(0));
    output_height = left_index.size(0);
  } else if (left_index_has_value && !right_index_has_value) {
    TORCH_CHECK(left_index.size(0) == other.size(0));
    output_height = left_index.size(0);
  } else if (!left_index_has_value && right_index_has_value) {
    TORCH_CHECK(right_index.size(0) == self.size(0));
    output_height = right_index.size(0);
  } else {
    TORCH_CHECK(self.size(0) == other.size(0));
    output_height = self.size(0);
  }

  auto op = get_binary_reduce_enum(reduce);

  auto output = at::empty({output_height, self.size(1)}, self.options());

  sampled_reduce_stub(
      kCPU,
      output,
      self.contiguous(),
      other.contiguous(),
      left_index,
      right_index,
      op);

  return output;
}

std::tuple<Tensor, Tensor> sampled_reduce_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other,
    const Tensor& left_index,
    const Tensor& right_index,
    c10::string_view reduce,
    std::array<bool, 2> output_mask) {

  bool left_index_has_value = left_index.defined();
  bool right_index_has_value = right_index.defined();

  auto op = get_binary_reduce_enum(reduce);
  const auto options = self.options();

  Tensor grad_left, grad_right;
  if (op == BinaryReductionType::ADD) {
    if (output_mask[0]) { grad_left = grad_output; }
    if (output_mask[1]) { grad_right = grad_output; }
  } else if (op == BinaryReductionType::SUB) {
    if (output_mask[0]) { grad_left = grad_output; }
    if (output_mask[1]) { grad_right = grad_output.neg(); }
  } else {
    if (output_mask[0]) { grad_left = at::empty(grad_output.sizes(), options); }
    if (output_mask[1]) { grad_right = at::empty(grad_output.sizes(), options); }

    sampled_reduce_backward_stub(
        kCPU,
        grad_left,
        grad_right,
        grad_output,
        self,
        other,
        left_index,
        right_index,
        op);
  }

  // expanded index to shape of `grad_output`,
  // this will trigger fast path on scatter_add
  auto _expanded = [&](const Tensor& t) {
    return t.as_strided(grad_output.sizes(), {1, 0});
  };

  Tensor grad_self, grad_other;
  if (output_mask[0]) {
    if (left_index_has_value) {
      grad_self = at::zeros(self.sizes(), options);
      grad_self.scatter_add_(/*dim*/0, _expanded(left_index), grad_left);
    } else {
      grad_self = grad_left.clone();
    }
  }
  if (output_mask[1]) {
    if (right_index_has_value) {
      grad_other = at::zeros(other.sizes(), options);
      grad_other.scatter_add_(/*dim*/0, _expanded(right_index), grad_right);
    } else {
      grad_other = grad_right.clone();
    }
  }

  return std::make_tuple(std::move(grad_self), std::move(grad_other));
}

DEFINE_DISPATCH(sampled_reduce_stub);
DEFINE_DISPATCH(sampled_reduce_backward_stub);

}} // namespace at::native
