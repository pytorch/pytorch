#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

using reduce_minmax_fn =
    void (*)(Tensor&, Tensor&, const Tensor&, int64_t, bool);

DECLARE_DISPATCH(reduce_minmax_fn, max_stub);
DECLARE_DISPATCH(reduce_minmax_fn, min_stub);
DECLARE_DISPATCH(reduce_minmax_fn, _aminmax_stub);

using where_fn = void (*)(TensorIterator &, ScalarType);
DECLARE_DISPATCH(where_fn, where_kernel);

using is_infinity_op_fn = void (*)(TensorIterator &);
DECLARE_DISPATCH(is_infinity_op_fn, isposinf_stub);
DECLARE_DISPATCH(is_infinity_op_fn, isneginf_stub);

using mode_fn = void (*)(Tensor&, Tensor&, const Tensor&, int64_t, bool);
DECLARE_DISPATCH(mode_fn, mode_stub);

}} // namespace at::native
