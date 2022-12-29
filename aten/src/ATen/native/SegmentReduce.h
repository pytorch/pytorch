#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>
#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/all.h>
#endif

namespace at { namespace native {

using segment_reduce_lengths_fn = Tensor (*)(
    ReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_lengths_fn, _segment_reduce_lengths_stub);

using segment_reduce_offsets_fn = Tensor (*)(
    ReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_offsets_fn, _segment_reduce_offsets_stub);

using segment_reduce_lengths_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    ReductionType,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_lengths_backward_fn, _segment_reduce_lengths_backward_stub);

using segment_reduce_offsets_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    ReductionType,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_offsets_backward_fn, _segment_reduce_offsets_backward_stub);

static void inline segment_reduce_check_inputs(
    const Tensor& data,
    const Tensor& lengths,
    const Tensor& indices,
    const Tensor& offsets,
    int64_t axis,
    bool unsafe) {
  TORCH_CHECK(data.numel() > 0,
      "segment_reduce: data can't be empty.");
  TORCH_CHECK(!indices.defined(),
      "segment_reduce: indices based reduction is not supported yet.");

  // check that one of lengths or offsets is defined
  bool offsets_has_value = offsets.defined();
  bool lengths_has_value = lengths.defined();
  TORCH_CHECK(lengths_has_value || offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.")

  const auto data_batches = data.sizes().slice(0, axis);

  if (offsets_has_value) {
    // offsets related checks
    TORCH_CHECK(data.get_device() == offsets.get_device(),
        "segment_reduce: Expected offsets and data on the same device");
    TORCH_CHECK(axis == offsets.dim() - 1,
        "segment_reduce: Expected axis to be the last dimension of offsets but got ",
        axis);
    TORCH_CHECK(offsets.size(axis) >= 2,
        "segment_reduce: Expected offsets.size(-1) >= 2 but got ",
        offsets.size(axis));

    // check batch sizes
    const auto offsets_batches = offsets.sizes().slice(0, axis);
    TORCH_CHECK(data_batches == offsets_batches,
        "segment_reduce: Expected data and offsets to have the same batch size, but got ",
        data_batches,
        " and ",
        offsets_batches);
  } else {
    // lengths related checks
    TORCH_CHECK(data.get_device() == lengths.get_device(),
        "segment_reduce: Expected lengths and data on the same device");
    TORCH_CHECK(axis == lengths.dim() - 1,
        "segment_reduce: Expected axis to be the last dimension of lengths but got ",
        axis);

    // check batch sizes
    const auto lengths_batches = lengths.sizes().slice(0, axis);
    TORCH_CHECK(data_batches == lengths_batches,
        "segment_reduce: Expected data and lengths to have the same batch size, but got ",
        data_batches,
        " and ",
        lengths_batches);

    if (!unsafe) {
      auto min_length = lengths.min().item<int64_t>();
      TORCH_CHECK((min_length >= 0),
          "segment_reduce: lengths contains negative value!");
      TORCH_CHECK(all(lengths.sum({-1}) == data.size(axis)).item<bool>(),
          "segment_reduce: Expected all rows of lengths along axis ",
          "to sum to data.size(lengths.dim()-1) when !unsafe.");
    }
  }
}

static inline void segment_reduce_check_inputs_backward(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    const Tensor& lengths,
    const Tensor& offsets,
    int64_t axis) {
  // check that one of lengths or offsets is defined
  auto lengths_has_value = lengths.defined();
  auto offsets_has_value = offsets.defined();
  TORCH_CHECK(lengths_has_value ||  offsets_has_value,
      "segment_reduce: Either lengths or offsets must be defined.");

  TORCH_CHECK(data.get_device() == grad.get_device(),
      "segment_reduce: Expected grad and data on the same device.");

  TORCH_CHECK(grad.sizes() == output.sizes(),
      "segment_reduce: Expected grad and output have the same sizes.");
}

}} // at::native
