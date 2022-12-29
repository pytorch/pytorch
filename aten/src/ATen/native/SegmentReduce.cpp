#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SegmentReduce.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

namespace at {
namespace native {

DEFINE_DISPATCH(_segment_reduce_lengths_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_stub);
DEFINE_DISPATCH(_segment_reduce_lengths_backward_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_backward_stub);

Tensor segment_reduce_kernel(
    const Tensor& data,
    c10::string_view reduce,
    const c10::optional<Tensor>& lengths_opt,
    const c10::optional<Tensor>& indices_opt,
    const c10::optional<Tensor>& offsets_opt,
    int64_t axis,
    bool unsafe,
    const c10::optional<Scalar>& initial_opt) {

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> lengths_maybe_owned = at::borrow_from_optional_tensor(lengths_opt);
  const Tensor& lengths = *lengths_maybe_owned;
  const Tensor& indices = c10::value_or_else(indices_opt, [] { return Tensor(); });
  const Tensor& offsets = c10::value_or_else(offsets_opt, [] { return Tensor(); });

  axis = maybe_wrap_dim(axis, data.ndimension());

  segment_reduce_check_inputs(
      data,
      lengths,
      indices,
      offsets,
      axis,
      unsafe);

  auto reduction = get_reduction_enum(reduce);

  if (offsets.defined()) {
    return _segment_reduce_offsets_stub(
      data.device().type(),
      reduction,
      data.contiguous(),
      offsets.contiguous(),
      axis,
      initial_opt);
  } else {
    return _segment_reduce_lengths_stub(
      data.device().type(),
      reduction,
      data.contiguous(),
      lengths.contiguous(),
      axis,
      initial_opt);
  }
}

// Currently some computation is being duplicated across forward and backward.
// TODO: Cache indices in forward pass to re-use in backward
Tensor _segment_reduce_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    c10::string_view reduce,
    const Tensor& lengths,
    const Tensor& offsets,
    int64_t axis,
    const c10::optional<Scalar>& initial_opt) {
  axis = maybe_wrap_dim(axis, data.ndimension());

  segment_reduce_check_inputs_backward(
      grad,
      output,
      data,
      lengths,
      offsets,
      axis);

  auto reduction = get_reduction_enum(reduce);

  if (offsets.defined()) {
    return _segment_reduce_offsets_backward_stub(
      grad.device().type(),
      grad.contiguous(),
      output.contiguous(),
      data.contiguous(),
      reduction,
      offsets.contiguous(),
      axis,
      initial_opt);
  } else {
    return _segment_reduce_lengths_backward_stub(
      grad.device().type(),
      grad.contiguous(),
      output.contiguous(),
      data.contiguous(),
      reduction,
      lengths.contiguous(),
      axis,
      initial_opt);
  }
}

} // namespace native
} // namespace at
