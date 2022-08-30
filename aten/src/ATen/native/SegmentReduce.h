#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

namespace at {
class Tensor;

namespace native {

enum SegmentReductionType { MAX, MEAN, MIN, SUM, PROD};

using segment_reduce_lengths_fn = Tensor (*)(
    SegmentReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_lengths_fn, _segment_reduce_lengths_stub);

using segment_reduce_offsets_fn = Tensor (*)(
    SegmentReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_offsets_fn, _segment_reduce_offsets_stub);

using segment_reduce_lengths_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    SegmentReductionType,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_lengths_backward_fn, _segment_reduce_lengths_backward_stub);

using segment_reduce_offsets_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    SegmentReductionType,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_offsets_backward_fn, _segment_reduce_offsets_backward_stub);

} // namespace native
} // namespace at
