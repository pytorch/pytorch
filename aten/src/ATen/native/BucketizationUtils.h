#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

template<typename F>
void searchsorted_generic_template(
  Tensor& result,
  const Tensor& input,
  const Tensor& boundaries,
  bool right,
  F&& contiguous_solver) {

  bool in_is_contiguous = input.is_contiguous();
  bool bd_is_contiguous = boundaries.is_contiguous();

  if (in_is_contiguous && bd_is_contiguous) {
    contiguous_solver(result, input, boundaries, right);
    return;
  }

  TORCH_WARN_ONCE("contiguous input tensors are expected, but got boundaries tensor ",
    (bd_is_contiguous ? "is contiguous" : "is not contiguous"), " and input value tensor ",
    (in_is_contiguous ? "is contiguous" : "is not contiguous"), " the non-contiguous tensors "
    "have been temporarily duplicated into contiguous tensors, this process has data copy between "
    "tensors which lower the performance, please make sure the input tensors are contiguous!");

  if (!in_is_contiguous && bd_is_contiguous) {
    Tensor input_cont = input.contiguous();
    contiguous_solver(result, input_cont, boundaries, right);
    return;
  }

  Tensor boundaries_cont = boundaries.contiguous();
  if (in_is_contiguous) {
    contiguous_solver(result, input, boundaries, right);
  }
  else {
    Tensor input_cont = input.contiguous();
    contiguous_solver(result, input_cont, boundaries_cont, right);
  }
}

bool dims_matched_before_last_dim(const Tensor& boundaries, const Tensor& input) {
  if (boundaries.dim() != input.dim()) {
    return false;
  }

  const auto& dims_bd = boundaries.sizes();
  const auto& dims_in = input.sizes();
  for (int64_t dim = 0; dim + 1 < boundaries.dim(); ++dim) {
    if (dims_bd[dim] != dims_in[dim]) {
      return false;
    }
  }
  return true;
}

void searchsorted_pre_check(const Tensor& boundaries, const Tensor& input, bool out_int32) {
  TORCH_CHECK(boundaries.device() == input.device(), "boundaries and input value tensors should have same device type, "
    "but we got boundaries tensor device type ", boundaries.device(), " and input value tensor device type ", input.device());

  TORCH_CHECK(boundaries.dtype() == input.dtype(), "boundaries and input value tensors should have same dtype, "
    "but we got boundaries tensor dtype ", boundaries.dtype(), " and input value tensor dtype ", input.dtype());

  TORCH_CHECK(boundaries.dim() != 0 && input.dim() != 0, "boundaries and input value tensors should have positive dimensions ",
    "but we got boundaries tensor dim(", boundaries.dim(), "), and input value tensor dim(", input.dim(), ")");

  TORCH_CHECK(boundaries.dim() == 1 || dims_matched_before_last_dim(boundaries, input),
    "boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor and input value tensor "
    "must match, but we got boundaries tensor ", boundaries.sizes(), " and input value tensor ", input.sizes());

  if (out_int32) {
    int numel_input = input.numel();
    TORCH_CHECK(numel_input <= INT_MAX && boundaries.sizes().back() <= INT_MAX,
      "total size of input and last dimension of boundaries should not exceed ", INT_MAX, " ",
      "but we got total size of input ", numel_input, " and last dimension of boundaries ", boundaries.sizes().back());
  }
}

}}
