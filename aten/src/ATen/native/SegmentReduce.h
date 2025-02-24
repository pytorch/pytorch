#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>
#include <c10/core/Scalar.h>
#include <optional>

namespace at {
class Tensor;

namespace native {

using segment_reduce_lengths_fn = Tensor (*)(
    ReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_lengths_fn, _segment_reduce_lengths_stub)

using segment_reduce_offsets_fn = Tensor (*)(
    ReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_offsets_fn, _segment_reduce_offsets_stub)

using segment_reduce_lengths_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    ReductionType,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_lengths_backward_fn, _segment_reduce_lengths_backward_stub)

using segment_reduce_offsets_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    ReductionType,
    const Tensor&,
    int64_t,
    const std::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_offsets_backward_fn, _segment_reduce_offsets_backward_stub)

} // namespace native
} // namespace at
