#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

using segment_reduce_fn =
    Tensor (*)(const Tensor&, const Tensor&, int64_t, bool);
DECLARE_DISPATCH(segment_reduce_fn, _segment_reduce_stub);

} // namespace native
} // namespace at
