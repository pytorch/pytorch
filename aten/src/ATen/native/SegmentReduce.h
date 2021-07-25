#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

enum SegmentReductionType { MAX, MEAN, MIN, SUM };

using segment_reduce_fn = Tensor (*)(
    SegmentReductionType,
    const Tensor&,
    const Tensor&,
    int64_t,
    const c10::optional<Scalar>&);
DECLARE_DISPATCH(segment_reduce_fn, _segment_reduce_stub);

using segment_reduce_backward_fn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    SegmentReductionType,
    const Tensor&,
    int64_t);
DECLARE_DISPATCH(segment_reduce_backward_fn, _segment_reduce_backward_stub);

} // namespace native
} // namespace at
