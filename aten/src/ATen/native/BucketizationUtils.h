#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

void searchsorted_maybe_trim_input_tensors(
  Tensor& trimmed_input,
  Tensor& trimmed_boundaries,
  const Tensor& raw_input,
  const Tensor& raw_boundaries) {

  bool in_is_contiguous = raw_input.is_contiguous();
  bool bd_is_contiguous = raw_boundaries.is_contiguous();

  if (!in_is_contiguous) {
    TORCH_WARN_ONCE("input value tensor is non-contiguous, this will lower the performance due to extra data copy "
      "when converting non-contiguous tensor to contiguous, please use contiguous input value tensor if possible");
    trimmed_input = raw_input.contiguous();
  }
  if (!bd_is_contiguous) {
    TORCH_WARN_ONCE("input value tensor is non-contiguous, this will lower the performance due to extra data copy "
      "when converting non-contiguous tensor to contiguous, please use contiguous input value tensor if possible");
    trimmed_boundaries = raw_boundaries.contiguous();
  }
  if (raw_input.dtype() != raw_boundaries.dtype()) {
    ScalarType input_stype = raw_input.scalar_type();
    ScalarType boundaries_stype = raw_boundaries.scalar_type();
    ScalarType common_stype = c10::promoteTypes(input_stype, boundaries_stype);
    if (common_stype != input_stype) {
      trimmed_input = in_is_contiguous ? raw_input.to(common_stype) : trimmed_input.to(common_stype);
    }
    if (common_stype != boundaries_stype) {
      trimmed_boundaries = bd_is_contiguous ? raw_boundaries.to(common_stype) : trimmed_boundaries.to(common_stype);
    }
  }
}

inline bool searchsorted_dims_matched_before_last_dim(const Tensor& boundaries, const Tensor& input) {
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

inline void searchsorted_pre_check(const Tensor& boundaries, const Tensor& input, const Tensor& output, bool out_int32) {
  TORCH_CHECK(boundaries.device() == input.device(), "boundaries and input value tensors should have same device type, ",
    "but we got boundaries tensor device type ", boundaries.device(), " and input value tensor device type ", input.device());

  TORCH_CHECK(input.dim() > 0 || (input.dim() == 0 && input.numel() == 1 && boundaries.dim() == 1),
    "input value can be a scalar only when boundaries tensor dimension is 1, but we got boundaries tensor ",
    "dim(", boundaries.dim(), ") and input value's dim(", input.dim(), ") numel(", input.numel(), ")");

  TORCH_CHECK(boundaries.dim() != 0, "boundaries tensor should have positive dimension, but got 0 dimension");

  TORCH_CHECK(boundaries.dim() == 1 || searchsorted_dims_matched_before_last_dim(boundaries, input),
    "boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor and input value tensor ",
    "must match, but we got boundaries tensor ", boundaries.sizes(), " and input value tensor ", input.sizes());

  ScalarType output_dtype = output.scalar_type();
  TORCH_CHECK((output_dtype == ScalarType::Long && !out_int32) || (output_dtype == ScalarType::Int && out_int32),
    "output tensor's dtype is wrong, it can only be Int(int32) or Long(int64) depending on whether out_int32 flag is True, ",
    "but we got output tensor's dtype ", output_dtype, " and out_int32 flag is ", (out_int32 ? "True" : "False"));

  if (out_int32) {
    int numel_input = input.numel();
    TORCH_CHECK(numel_input <= INT_MAX && boundaries.sizes().back() <= INT_MAX,
      "total size of input and last dimension of boundaries should not exceed ", INT_MAX, " ",
      "but we got total size of input ", numel_input, " and last dimension of boundaries ", boundaries.sizes().back());
  }
}

}}
