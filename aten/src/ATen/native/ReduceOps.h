#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
class Tensor;
}

namespace at { namespace native {

using reduce_fn = void(*)(TensorIterator &);

DECLARE_DISPATCH(reduce_fn, sum_stub);
DECLARE_DISPATCH(reduce_fn, nansum_stub);
DECLARE_DISPATCH(reduce_fn, prod_stub);
DECLARE_DISPATCH(reduce_fn, mean_stub);
DECLARE_DISPATCH(reduce_fn, and_stub);
DECLARE_DISPATCH(reduce_fn, or_stub);
DECLARE_DISPATCH(reduce_fn, min_values_stub);
DECLARE_DISPATCH(reduce_fn, max_values_stub);
DECLARE_DISPATCH(reduce_fn, argmax_stub);
DECLARE_DISPATCH(reduce_fn, argmin_stub);

using reduce_std_var_function =
    void (*)(TensorIterator&, int64_t correction, bool take_sqrt);
DECLARE_DISPATCH(reduce_std_var_function, std_var_stub);

using reduce_norm_fn =
    void (*)(Tensor&, const Tensor&, const c10::Scalar&, c10::optional<int64_t>);
DECLARE_DISPATCH(reduce_norm_fn, norm_kernel);

using reduce_fn_flag = void(*)(TensorIterator &, const c10::Scalar&);
DECLARE_DISPATCH(reduce_fn_flag, norm_stub);

using structured_cum_fn = void (*)(const Tensor&, const Tensor&, int64_t);
using cum_fn = void (*)(Tensor&, const Tensor&, int64_t);
DECLARE_DISPATCH(structured_cum_fn, cumsum_stub);
DECLARE_DISPATCH(structured_cum_fn, cumprod_stub);
DECLARE_DISPATCH(cum_fn, logcumsumexp_stub);

DECLARE_DISPATCH(void (*)(const Tensor&, int64_t, bool, Tensor&, Tensor&), aminmax_stub);
DECLARE_DISPATCH(void (*)(const Tensor&, Tensor&, Tensor&), aminmax_allreduce_stub);

// Used in cuda/Normalization.cu
TORCH_API std::tuple<Tensor&,Tensor&> var_mean_out(
    Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim,
    int64_t correction, bool keepdim);

}} // namespace at::native
