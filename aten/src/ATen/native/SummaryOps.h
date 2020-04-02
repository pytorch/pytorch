#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {

struct TensorIterator;

namespace native {

using hist_fn = void (*)(TensorIterator&, Tensor&, int64_t, Scalar, Scalar);
DECLARE_DISPATCH(hist_fn, histc_stub);

} // namespace native

} // namespace at
