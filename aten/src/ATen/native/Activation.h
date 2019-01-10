#pragma once

#include <c10/core/Scalar.h>
#include <ATen/native/DispatchStub.h>

namespace at { struct TensorIterator; }

namespace at { namespace native {

using threshold_fn = void(*)(TensorIterator&, Scalar, Scalar);

DECLARE_DISPATCH(threshold_fn, threshold_stub);


}} // namespace at::native
