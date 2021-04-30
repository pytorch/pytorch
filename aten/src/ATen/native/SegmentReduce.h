#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

using segment_reduce_fn =
    Tensor (*)(const Tensor&, const Tensor&, int64_t, bool);
DECLARE_DISPATCH(segment_reduce_fn, _segment_reduce_stub);

using segment_reduce_backward_fn =
    Tensor (*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
DECLARE_DISPATCH(segment_reduce_backward_fn, _segment_reduce_backward_stub);

} // namespace native
} // namespace at
