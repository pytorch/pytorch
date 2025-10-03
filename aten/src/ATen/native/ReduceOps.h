#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/OptionalArrayRef.h>
#include <optional>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
class Tensor;
struct Dimname;
using DimnameList = c10::ArrayRef<Dimname>;
using OptionalIntArrayRef = c10::OptionalArrayRef<int64_t>;
}

namespace at::native {

using reduce_fn = void(*)(TensorIterator &);

DECLARE_DISPATCH(reduce_fn, sum_stub)
DECLARE_DISPATCH(reduce_fn, nansum_stub)
DECLARE_DISPATCH(reduce_fn, prod_stub)
DECLARE_DISPATCH(reduce_fn, mean_stub)
DECLARE_DISPATCH(reduce_fn, and_stub)
DECLARE_DISPATCH(reduce_fn, or_stub)
DECLARE_DISPATCH(reduce_fn, min_values_stub)
DECLARE_DISPATCH(reduce_fn, max_values_stub)
DECLARE_DISPATCH(reduce_fn, argmax_stub)
DECLARE_DISPATCH(reduce_fn, argmin_stub)
DECLARE_DISPATCH(reduce_fn, xor_sum_stub)

using reduce_std_var_function =
    void (*)(TensorIterator&, double correction, bool take_sqrt);
DECLARE_DISPATCH(reduce_std_var_function, std_var_stub)

using reduce_norm_fn =
    void (*)(Tensor&, const Tensor&, const c10::Scalar&, std::optional<int64_t>);
DECLARE_DISPATCH(reduce_norm_fn, norm_kernel)

using reduce_fn_flag = void(*)(TensorIterator &, const c10::Scalar&);
DECLARE_DISPATCH(reduce_fn_flag, norm_stub)

using structured_cum_fn = void (*)(const Tensor&, const Tensor&, int64_t);
using cum_fn = void (*)(Tensor&, const Tensor&, int64_t);
DECLARE_DISPATCH(structured_cum_fn, cumsum_stub)
DECLARE_DISPATCH(structured_cum_fn, cumprod_stub)
DECLARE_DISPATCH(cum_fn, logcumsumexp_stub)

DECLARE_DISPATCH(void (*)(const Tensor&, int64_t, bool, Tensor&, Tensor&), aminmax_stub)
DECLARE_DISPATCH(void (*)(const Tensor&, Tensor&, Tensor&), aminmax_allreduce_stub)

// Used in cuda/Normalization.cu
TORCH_API std::tuple<Tensor&,Tensor&> var_mean_out(
    Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim,
    int64_t correction, bool keepdim);

// Forward declarations for logsumexp dispatchers (defined in ReduceOps.cpp)
// These are called by codegen for CompositeExplicitAutograd dispatch
TORCH_API Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim);
TORCH_API Tensor logsumexp(const Tensor& self);
TORCH_API Tensor logsumexp(const Tensor& self, OptionalIntArrayRef opt_dims, bool keepdim);
TORCH_API Tensor logsumexp(const Tensor& self, DimnameList dims, bool keepdim);
// Note: logsumexp_out(self, result) is auto-generated for the no-dim case
TORCH_API Tensor& logsumexp_out(const Tensor& self, DimnameList dims, bool keepdim, Tensor& result);

} // namespace at::native
