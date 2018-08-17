#pragma once

#include <ATen/ATen.h>
#include <ATen/core/optional.h>
#include <ATen/native/DispatchStub.h>

namespace at {
  struct TensorIterator;
}

namespace at { namespace native {

using reduce_fn = void(*)(TensorIterator &);

DECLARE_DISPATCH(reduce_fn, sum_stub);
DECLARE_DISPATCH(reduce_fn, prod_stub);

using reduce_norm_fn = void(*)(Tensor &, const Tensor &, Scalar, at::optional<int64_t>);
DECLARE_DISPATCH(reduce_norm_fn, norm_kernel);

}} // namespace at::native
