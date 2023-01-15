#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/result_type.h>
#endif

namespace at {
namespace native {

// original values given by raw_*. If an original value is not contiguous, will make a contiguous copy to
// the corresponding trimmed_* value. Additionally, if the dtypes of the boundary and input tensor do not
// match, will change them to be a common super type so comparisons are done between the same types.
// For any trimmed_* tensor, if its outgoing value matches what it was incoming (typically null), then the
// corresponding raw_* version should be used since it was already contiguous of the right type.
inline void searchsorted_maybe_trim_input_tensors(
    Tensor& trimmed_input,
    Tensor& trimmed_boundaries,
    Tensor& trimmed_sorter,
    const Tensor& raw_input,
    const Tensor& raw_boundaries,
    const Tensor& raw_sorter) {
  bool in_is_contiguous = raw_input.is_contiguous();
  bool bd_is_contiguous = raw_boundaries.is_contiguous();
  bool sort_is_contiguous = raw_sorter.is_contiguous();

  if (!in_is_contiguous) {
    TORCH_WARN_ONCE("torch.searchsorted(): input value tensor is non-contiguous, this will lower the performance due "
      "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous input value "
      "tensor if possible. This message will only appear once per program.");
    trimmed_input = raw_input.contiguous();
  }
  if (!bd_is_contiguous) {
    TORCH_WARN_ONCE("torch.searchsorted(): boundary tensor is non-contiguous, this will lower the performance due "
      "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous boundary "
      "tensor if possible. This message will only appear once per program.");
    trimmed_boundaries = raw_boundaries.contiguous();
  }
  if (!sort_is_contiguous) {
    TORCH_WARN_ONCE("torch.searchsorted(): sorter tensor is non-contiguous, this will lower the performance due "
      "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous sorter "
      "tensor if possible. This message will only appear once per program.");
    trimmed_sorter = raw_sorter.contiguous();
  }
  if (raw_input.dtype() != raw_boundaries.dtype()) {
    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(raw_boundaries, state);
    state = at::native::update_result_type_state(raw_input, state);
    ScalarType common_stype = at::native::result_type(state);

    TORCH_INTERNAL_ASSERT(common_stype != ScalarType::Undefined);
    if (common_stype != raw_input.scalar_type()) {
      trimmed_input = in_is_contiguous ? raw_input.to(common_stype) : trimmed_input.to(common_stype);
    }
    if (common_stype != raw_boundaries.scalar_type()) {
      trimmed_boundaries = bd_is_contiguous ? raw_boundaries.to(common_stype) : trimmed_boundaries.to(common_stype);
    }
  }
}

/* unused but needed for internal jagged tensor class */
inline void searchsorted_maybe_trim_input_tensors(
    Tensor& trimmed_input,
    Tensor& trimmed_boundaries,
    const Tensor& raw_input,
    const Tensor& raw_boundaries) {
  Tensor trimmed_sorter;
  Tensor raw_sorter;
  return searchsorted_maybe_trim_input_tensors(
      trimmed_input,
      trimmed_boundaries,
      trimmed_sorter,
      raw_input,
      raw_boundaries,
      raw_sorter);
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

inline Tensor searchsorted_scalar_tensor(const Scalar& scalar, const c10::Device& device) {
  auto tensor = c10::scalar_to_tensor(scalar, device);
  // This is to adopt the scalar promotion rules defined in native/TypeProperties.h
  // So we have the same type promotion rules as binary operations.
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

inline void searchsorted_pre_check(
    const Tensor& boundaries,
    const Tensor& input,
    const Tensor& output,
    const bool out_int32,
    const bool right,
    const c10::optional<c10::string_view> side_opt,
    const Tensor& sorter) {
  if (side_opt) {
    const c10::string_view side = *side_opt;
    TORCH_CHECK(side == "left" || side == "right", "torch.searchsorted(): side can only be 'left' or 'right' but ",
      "got ", side);

    // assume the user has not explicitly set (right=False, side="right")
    TORCH_CHECK(!right || side == "right", "torch.searchsorted(): side and right can't be set to opposites, got side "
    "of ", side, " while right was True");
  }

  TORCH_CHECK(boundaries.device() == input.device(), "torch.searchsorted(): boundaries and input value tensors ",
    "should have same device type, but got boundaries tensor device type ", boundaries.device(), " and input value ",
    "tensor device type ", input.device());

  if (sorter.defined()) {
    TORCH_CHECK(sorter.device() == boundaries.device(), "torch.searchsorted(): sorter and boundary tensors should ",
      "have same device type, but got sorter tensor device type ", sorter.device(), " and input value tensor ",
      "device type ", boundaries.device());

    TORCH_CHECK(sorter.sizes() == boundaries.sizes(), "torch.searchsorted(): boundary and sorter must have the same "
      "size, but got boundary tensor ", boundaries.sizes(), "and got sorter tensor ", sorter.sizes());

    TORCH_CHECK(sorter.scalar_type() == ScalarType::Long, "torch.searchsorted(): sorter must be a tensor of long ",
      "dtype but got dtype ", sorter.scalar_type());
  }

  TORCH_CHECK(input.dim() > 0 || (input.dim() == 0 && input.numel() == 1 && boundaries.dim() == 1),
    "torch.searchsorted(): input value can be a scalar only when boundaries tensor dimension is 1, but we got ",
    "boundaries tensor dim(", boundaries.dim(), ") and input value's dim(", input.dim(), ") numel(",
    input.numel(), ")");

  TORCH_CHECK(boundaries.dim() != 0, "torch.searchsorted(): boundaries tensor should have positive dimension, but ",
    "got 0 dimension");

  TORCH_CHECK(boundaries.dim() == 1 || searchsorted_dims_matched_before_last_dim(boundaries, input),
    "torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor ",
    "and input value tensor must match, but we got boundaries tensor ", boundaries.sizes(), " and input value tensor ",
    input.sizes());

  ScalarType output_dtype = output.scalar_type();
  TORCH_CHECK(
      (output_dtype == ScalarType::Long && !out_int32) ||
          (output_dtype == ScalarType::Int && out_int32),
      "torch.searchsorted(): output tensor's dtype is wrong, it can only be Int(int32) or Long(int64) depending on ",
      "whether out_int32 flag is True, but we got output tensor's dtype ", output_dtype,
      " and out_int32 flag is ", (out_int32 ? "True" : "False"));

  if (out_int32) {
    TORCH_CHECK(boundaries.sizes().back() < INT_MAX,
      "torch.searchsorted(): the size of boundaries' last dimension should be less than ", INT_MAX, ", but we got ",
      boundaries.sizes().back());
  }
}

}}
