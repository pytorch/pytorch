#pragma once

#include <ATen/native/DispatchStub.h>

namespace c10 {
class Scalar;
}

namespace at {
class Tensor;
struct TensorIterator;
struct TensorIteratorBase;
} // namespace at

namespace at::native {

using reduce_minmax_fn =
    void (*)(Tensor&, Tensor&, const Tensor&, int64_t, bool);
using structured_reduce_minmax_fn =
    void (*)(const Tensor&, const Tensor&, const Tensor&, int64_t, bool);

DECLARE_DISPATCH(structured_reduce_minmax_fn, max_stub)
DECLARE_DISPATCH(structured_reduce_minmax_fn, min_stub)

using where_fn = void (*)(TensorIterator&);
DECLARE_DISPATCH(where_fn, where_kernel)

using is_infinity_op_fn = void (*)(TensorIteratorBase&);
DECLARE_DISPATCH(is_infinity_op_fn, isposinf_stub)
DECLARE_DISPATCH(is_infinity_op_fn, isneginf_stub)

using mode_fn = void (*)(Tensor&, Tensor&, const Tensor&, int64_t, bool);
DECLARE_DISPATCH(mode_fn, mode_stub)

using clamp_tensor_fn = void (*)(TensorIteratorBase&);
DECLARE_DISPATCH(clamp_tensor_fn, clamp_stub)

namespace detail {
enum class ClampLimits { Min, Max, MinMax };
}

DECLARE_DISPATCH(
    void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&),
    clamp_scalar_stub)
DECLARE_DISPATCH(
    void (*)(TensorIteratorBase&, c10::Scalar),
    clamp_min_scalar_stub)
DECLARE_DISPATCH(
    void (*)(TensorIteratorBase&, c10::Scalar),
    clamp_max_scalar_stub)

using isin_default_fn =
    void (*)(const Tensor&, const Tensor&, bool, const Tensor&);
DECLARE_DISPATCH(isin_default_fn, isin_default_stub)

} // namespace at::native
