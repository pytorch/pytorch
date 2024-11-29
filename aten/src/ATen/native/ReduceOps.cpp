#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ReduceOps.h>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorDimApply.h>
#include <ATen/core/grad_mode.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cummax_helper.h>
#include <ATen/ops/_cummax_helper_native.h>
#include <ATen/ops/_cummin_helper.h>
#include <ATen/ops/_cummin_helper_native.h>
#include <ATen/ops/_is_all_true_native.h>
#include <ATen/ops/_is_any_true_native.h>
#include <ATen/ops/_logcumsumexp.h>
#include <ATen/ops/_logcumsumexp_native.h>
#include <ATen/ops/_sparse_csr_sum.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_sum_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/add.h>
#include <ATen/ops/all_meta.h>
#include <ATen/ops/all_native.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/amax_meta.h>
#include <ATen/ops/amax_native.h>
#include <ATen/ops/amin_meta.h>
#include <ATen/ops/amin_native.h>
#include <ATen/ops/aminmax_meta.h>
#include <ATen/ops/aminmax_native.h>
#include <ATen/ops/any_meta.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/argmax_meta.h>
#include <ATen/ops/argmax_native.h>
#include <ATen/ops/argmin_meta.h>
#include <ATen/ops/argmin_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/cummax.h>
#include <ATen/ops/cummax_native.h>
#include <ATen/ops/cummaxmin_backward_native.h>
#include <ATen/ops/cummin.h>
#include <ATen/ops/cummin_native.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/cumprod_backward_native.h>
#include <ATen/ops/cumprod_meta.h>
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/cumsum_meta.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/diff_native.h>
#include <ATen/ops/dist_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/equal_native.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/gradient_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/logcumsumexp.h>
#include <ATen/ops/logcumsumexp_native.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/logsumexp_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mean_meta.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/nanmean_native.h>
#include <ATen/ops/nansum.h>
#include <ATen/ops/nansum_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/native_norm.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/norm_meta.h>
#include <ATen/ops/norm_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/prod_meta.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/special_logsumexp_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/std.h>
#include <ATen/ops/std_mean.h>
#include <ATen/ops/std_mean_native.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_meta.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/value_selecting_reduction_backward_native.h>
#include <ATen/ops/var.h>
#include <ATen/ops/var_mean.h>
#include <ATen/ops/var_mean_native.h>
#include <ATen/ops/var_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <c10/util/irange.h>
#include <c10/util/SmallBuffer.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace at::meta {

static ScalarType infer_dtype_from_optional(
    const Tensor& self,
    const std::optional<ScalarType>& opt_dtype,
    const Tensor& result) {
  // 'opt_dtype' has the priority for both cases.
  if (result.defined()) {
    // Otherwise, get the result type, if defined.
    return opt_dtype.value_or(result.scalar_type());
  } else {
    // Last case is to get the self type.
    // If the self type is an integer, we promote it to kLong.
    return at::native::get_dtype_from_self(self, opt_dtype, true);
  }
}

static IntArrayRef optional_to_arrayref(const std::optional<int64_t>& opt) {
  return opt.has_value() ? opt.value() : IntArrayRef{};
}

static ScalarType get_result_or_bytebool_dtype(const Tensor& self, const Tensor& result) {
  // Refer [all, any : uint8 compatibility]
  if (result.defined()) {
    return result.scalar_type();
  } else {
    return (self.scalar_type() == kByte) ? kByte : kBool;
  }
}

static void check_result_is_bytebool(const char* name, const Tensor& self, const Tensor& result) {
  if (result.defined()) {
    // Refer [all, any : uint8 compatibility]
    TORCH_CHECK(
        result.scalar_type() == ScalarType::Bool ||
            result.scalar_type() == ScalarType::Byte,
        name, " only supports bool tensor for result, got: ",
        result.scalar_type());
  }
}

// Note [all, any : uint8 compatibility]:
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For NumPy compatibility, `all` and `any` return
// Tensor of dtype `bool`. However for compatibility reason,
// for `uint8`, they return Tensor of same dtype `uint8`.
// Reference: https://github.com/pytorch/pytorch/pull/47878#issuecomment-747108561
static void allany_meta(
    impl::MetaBase& meta,
    const char* name,
    const Tensor& self,
    OptionalIntArrayRef dims,
    bool keepdim) {
  const auto& result = meta.maybe_get_output();
  check_result_is_bytebool(name, self, result);
  auto out_dtype = get_result_or_bytebool_dtype(self, result);
  resize_reduction(meta, self, dims, keepdim, out_dtype, /*allow_empty_dims=*/true);
}

TORCH_META_FUNC2(all, dim)(const Tensor& self, int64_t dim, bool keepdim) {
  allany_meta(*this, "all", self, dim, keepdim);
}

TORCH_META_FUNC2(all, dims)(const Tensor& self, OptionalIntArrayRef dim, bool keepdim) {
  allany_meta(*this, "all", self, dim, keepdim);
}

TORCH_META_FUNC(all)(const Tensor& self) {
  allany_meta(*this, "all", self, {}, false);
}

TORCH_META_FUNC2(any, dim)(const Tensor& self, int64_t dim, bool keepdim) {
  allany_meta(*this, "any", self, dim, keepdim);
}

TORCH_META_FUNC2(any, dims)(const Tensor& self, OptionalIntArrayRef dim, bool keepdim) {
  allany_meta(*this, "any", self, dim, keepdim);
}

TORCH_META_FUNC(any)(const Tensor& self) {
  allany_meta(*this, "any", self, {}, false);
}

static void check_argmax_argmin(
    const char* name,
    const Tensor& self,
    const std::optional<int64_t>& dim) {
  if (dim.has_value()) {
    auto dim_ = maybe_wrap_dim(dim.value(), self.dim());
    native::zero_numel_check_dims(self, dim_, name);
  } else {
    TORCH_CHECK_INDEX(
        self.numel() != 0,
        name, ": Expected reduction dim to be specified for input.numel() == 0.");
  }
}

TORCH_META_FUNC(argmax)
(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
  check_argmax_argmin("argmax()", self, dim);
  resize_reduction(*this, self, optional_to_arrayref(dim), keepdim, kLong);
}

TORCH_META_FUNC(argmin)
(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
  check_argmax_argmin("argmin()", self, dim);
  resize_reduction(*this, self, optional_to_arrayref(dim), keepdim, kLong);
}

static void meta_func_cum_ops(
    impl::MetaBase& meta,
    const char* name,
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
  // Checking whether 'dim' is valid.
  maybe_wrap_dim(dim, self.dim());

  const auto& result = meta.maybe_get_output();
  ScalarType out_dtype{};

  if (result.defined()) {
    out_dtype = dtype.value_or(result.scalar_type());
  } else {
    auto is_integral = at::isIntegralType(self.scalar_type(), /*includeBool=*/true);
    out_dtype = dtype.value_or(is_integral ? ScalarType::Long : self.scalar_type());
  }

  meta.set_output_raw_strided(0, self.sizes(), {}, self.options().dtype(out_dtype));
  namedinference::propagate_names(result, self);
}

TORCH_META_FUNC(cumsum)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype) {
  meta_func_cum_ops(*this, "cumsum", self, dim, dtype);
}

TORCH_META_FUNC(cumprod)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype) {
  meta_func_cum_ops(*this, "cumprod", self, dim, dtype);
}

TORCH_META_FUNC2(sum, dim_IntList)
(const Tensor& self, OptionalIntArrayRef opt_dim, bool keepdim, std::optional<ScalarType> opt_dtype) {
  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
  resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}

TORCH_META_FUNC2(prod, dim_int)
(const Tensor& self,
 int64_t dim,
 bool keepdim,
 std::optional<ScalarType> dtype) {
  auto out_dtype = infer_dtype_from_optional(self, dtype, maybe_get_output());
  resize_reduction(*this, self, dim, keepdim, out_dtype);
}

TORCH_META_FUNC2(mean, dim)
(const Tensor& self, OptionalIntArrayRef opt_dim, bool keepdim, std::optional<ScalarType> opt_dtype) {
  auto in_dtype = at::native::get_dtype_from_self(self, opt_dtype, true);

  if (!at::isFloatingType(in_dtype) && !at::isComplexType(in_dtype)) {
    std::string what = "Input";
    std::string dtype = toString(self.scalar_type());

    if (opt_dtype.has_value()) {
      what = "Optional";
      dtype = toString(opt_dtype.value());
    }

    TORCH_CHECK(
        false,
        "mean(): could not infer output dtype. ",
        what, " dtype must be either a floating point or complex dtype. ",
        "Got: ", dtype);
  }

  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
  resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}

static ScalarType get_result_or_self_value_dtype(
    const Tensor& self,
    const Tensor& result,
    const std::optional<ScalarType>& dtype) {
  if (result.defined()) {
    return result.scalar_type();
  } else {
    return dtype.value_or(toRealValueType(self.scalar_type()));
  }
}

TORCH_META_FUNC2(norm, ScalarOpt_dim)
(const Tensor& self, const OptionalScalarRef p, IntArrayRef dim, bool keepdim) {
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
      "norm(): input dtype should be either floating point or complex. "
      "Got ", self.scalar_type(), " instead.");

  auto out_dtype = get_result_or_self_value_dtype(self, maybe_get_output(), std::nullopt);
  resize_reduction(*this, self, dim, keepdim, out_dtype);
}

TORCH_META_FUNC2(norm, ScalarOpt_dim_dtype)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 ScalarType dtype) {
  TORCH_CHECK(
      at::isFloatingType(dtype) || at::isComplexType(dtype),
      "norm(): the desired output dtype should be either floating point or complex. "
      "Got ", dtype, " instead.");

  auto out_dtype = get_result_or_self_value_dtype(self, maybe_get_output(), dtype);
  resize_reduction(*this, self, dim, keepdim, out_dtype);
}

TORCH_META_FUNC(aminmax)
(const Tensor& self, std::optional<int64_t> dim_opt, bool keepdim) {
  DimVector shape;
  if (dim_opt.has_value()) {
    auto dim = maybe_wrap_dim(dim_opt.value(), self.ndimension());
    native::zero_numel_check_dims(self, dim, "aminmax");
    shape = get_reduction_shape(self, dim, keepdim);
  } else {
    TORCH_CHECK(
        self.numel() > 0,
        "aminmax(): cannot compute aminmax over an empty dimension as the "
        "operation has no identity.");
    if (keepdim) {
      shape = DimVector(self.ndimension(), 1);
    }
  }
  const auto options = self.options();
  this->set_output_raw_strided(0, shape, {}, options);
  this->set_output_raw_strided(1, shape, {}, options);
}

TORCH_META_FUNC(amax)
(const Tensor& self, IntArrayRef dim, bool keepdim) {
  auto maybe_result = maybe_get_output();
  if (maybe_result.defined()) {
    TORCH_CHECK(self.scalar_type() == maybe_result.scalar_type(), "Expected the dtype for input and out to match, but got ",
            self.scalar_type(), " for input's dtype and ",  maybe_result.scalar_type(), " for out's dtype.");
  }
  if (self.numel() == 0) {
    at::native::zero_numel_check_dims(self, dim, "amax()");
  }
  const ScalarType& out_dtype = maybe_result.defined() ? maybe_result.scalar_type() : self.scalar_type();
  resize_reduction(*this, self, dim, keepdim, out_dtype);
}

TORCH_META_FUNC(amin)
(const Tensor& self, IntArrayRef dim, bool keepdim) {
  auto maybe_result = maybe_get_output();
  if (maybe_result.defined()) {
    TORCH_CHECK(self.scalar_type() == maybe_result.scalar_type(), "Expected the dtype for input and out to match, but got ",
                self.scalar_type(), " for input's dtype and ",  maybe_result.scalar_type(), " for out's dtype.");
  }
  if (self.numel() == 0) {
    at::native::zero_numel_check_dims(self, dim, "amin()");
  }
  const ScalarType& out_dtype = maybe_result.defined() ? maybe_result.scalar_type() : self.scalar_type();
  resize_reduction(*this, self, dim, keepdim, out_dtype);
}

} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(aminmax_stub);
DEFINE_DISPATCH(aminmax_allreduce_stub);

TORCH_IMPL_FUNC(aminmax_out)
(const Tensor& self,
 std::optional<int64_t> dim_opt,
 bool keepdim,
 const Tensor& min,
 const Tensor& max) {
  auto mutable_min = const_cast<Tensor&>(min);
  auto mutable_max = const_cast<Tensor&>(max);
  if (dim_opt.has_value()) {
    aminmax_stub(
        self.device().type(),
        self,
        maybe_wrap_dim(dim_opt.value(), self.ndimension()),
        keepdim,
        mutable_min,
        mutable_max);
  } else {
    aminmax_allreduce_stub(self.device().type(), self.contiguous(), mutable_min, mutable_max);
  }
}

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(nansum_stub);
DEFINE_DISPATCH(std_var_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(norm_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(and_stub);
DEFINE_DISPATCH(or_stub);
DEFINE_DISPATCH(min_values_stub);
DEFINE_DISPATCH(max_values_stub);
DEFINE_DISPATCH(argmax_stub);
DEFINE_DISPATCH(argmin_stub);
DEFINE_DISPATCH(cumsum_stub);
DEFINE_DISPATCH(cumprod_stub);
DEFINE_DISPATCH(logcumsumexp_stub);

Tensor _logcumsumexp_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_cpu(self, dim, result);
}

Tensor& _logcumsumexp_out_cpu(const Tensor& self, int64_t dim, Tensor& result) {
  logcumsumexp_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor logcumsumexp(const Tensor& self, int64_t dim) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_logcumsumexp(self, dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& logcumsumexp_out(const Tensor& self, int64_t dim, Tensor& result) {
  check_scalar_type_device_layout_equal(result, self);
  {
    NoNamesGuard guard;
    at::_logcumsumexp_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

template <class Stub>
void impl_func_cum_ops(
    const Tensor& self,
    int64_t dim,
    const Tensor& result,
    Stub& stub) {
  NoNamesGuard guard;
  if (self.dim() == 0) {
    result.fill_(self);
  } else if (self.numel() == 0) {
    result.zero_();
  } else {
    dim = maybe_wrap_dim(dim, self.dim());
    stub(self.device().type(), result, self.to(result.scalar_type()), dim);
  }
}

TORCH_IMPL_FUNC(cumsum_out)
(const Tensor& self,
 int64_t dim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  impl_func_cum_ops(self, dim, result, cumsum_stub);
}

TORCH_IMPL_FUNC(cumprod_out)
(const Tensor& self,
 int64_t dim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  impl_func_cum_ops(self, dim, result, cumprod_stub);
}

static Tensor reversed_cumsum(const Tensor& w, int64_t dim) {
  return w.flip(dim).cumsum(dim).flip(dim);
}

Tensor cumprod_backward(const Tensor& grad, const Tensor& input, int64_t dim, const Tensor& output) {
  /*
    We show here how to derive an O(n) gradient formula for
    arbitrary inputs. It follows via a basic application of the
    chain rule together with a number of observations for different
    cases. We assume that x is an n-dimensional vector and y = cumprod(x).
    In the actual implementation we will need to play a bit with masks
    to be able to implement the formulas deduced here for tensors.

    We will first deduce the formula for the case when
    x[i] != 0 for 1 <= i <= n.

    For F : R^n -> R the cost function (we will look at the complex case later),
    we have

    dF / dx_k = sum_j (dF / dy_j) * (dy_j / dx_k)   (1)

    The term dF / dy_j is just grad_output[j] (assuming again
    everything is one-dimensional).

    The term (dy_j / dx_k) is easily seen to be

    if j >= k
      dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i
    else:
      dy_j / dx_k = 0

    Note that the indicator (j>=k) can be taken out
    by replacing the sum in (1) with a sum from
    k <= j <= n.

    Thus,
    dF / dx_k = sum_{k <= j <= n} grad_output[j] * (dy_j / dx_k)

    with
    dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i     (2)

    Note that this last term is just the cumulative product
    with k omitted. Thus, if x_k (the input) is nonzero, we can
    just express this as

    dy_j / dx_k = (prod_{1 <= i <= j} x_i) / x_k
                = y_j / x_k

    So therefore,

    dF / dx_k = sum_{k <= j <= n} grad_output[j] * y_j / x_k

    This formula just makes sense when input[i] != 0 for every i.

    Assume now that there exists at least a zero in the input.
    Denote by z1 the first element 1 <= z1 <= n with input[z1] = 0
    and z2 the second element z1 < z2 <= n with input[z2] = 0,
    (or z2 = n if there is just one zero in input)

    We have three cases.

    k > z1:
    Looking at (2), we see that dy_j / dx_k = 0, for j >= k, as these terms
    all include a x_{z1} which is zero. As such, dF / dx_k = 0 in this case

    k < z1:
    Reasoning as in the previous case, we see that for these elements we have that

    dF / dx_k = sum_{k <= j < z1} grad_output[j] * (dy_j / dx_k)

    as the terms of the sum for j in z1 <= j <= n are all zero

    k = z1:
    Similar to the case k < z1, we have that

    dF / dx_z1 = sum_{z1 <= j < z2} grad_output[j] * (dy_j / dx_z1)

    This case has a subtlety though. To compute (dy_j / dx_z1), we cannot use the formula

    dy_j / dx_z1 = y_j / x_z1

    as, y_j = x_z1 = 0 for j >= z1. We need to compute it with the formula for its derivative,
    that is:

    dy_j / dx_z1 = prod(x[:z1]) * (grad_output[z1] + sum(grad_output[z1+1:z2] * cumprod(x[z1+1:z2])))

    When the inputs are complex, this is map is holomorphic. As such, to compute
    its backwards is just the conjugate of the usual backwards. This simplifies to
    conjugating the input. We may also reuse the output as, since the map is holomorphic,
    cumprod(input.conj()) = cumprod(input).conj()
  */

  if (input.sym_numel() <= 1) {
    return grad;
  }
  dim = at::maybe_wrap_dim(dim, input.dim());
  const int64_t dim_size = input.sym_sizes()[dim].guard_int(__FILE__, __LINE__);
  if (dim_size == 1) {
    return grad;
  }

  // To enable complex support.
  // From this line on `input_conj` and output_conj`
  // are interchangeable with `input` and `output`.
  auto input_conj = input.conj();
  auto output_conj = output.conj();

  // For Composite Compliance, we always choose the slower but composite compliant path.
  bool are_inputs_tensors_sublcass = areAnyTensorSubclassLike({input, grad, output});

  const auto w = output_conj * grad;
  const auto is_zero = input == 0;
  if (!are_inputs_tensors_sublcass) {
    if (is_zero.any().item<uint8_t>() == 0) {
      return reversed_cumsum(w, dim).div(input_conj);
    }
  }

  // If we are not computing a second order gradient, we can use an
  // O(n) implementation. The derivative of this implementation is _not_
  // the second derivative of cumprod. As such, we fallback to a less efficient
  // O(n^2) implementation when at::GradMode::is_enabled().
  if (!at::GradMode::is_enabled() && !are_inputs_tensors_sublcass) {
    // n.b. This could probably be implemented much faster with a kernel

    // From here on we need to use some mask gymnastics to
    // account for the tensorial dimensions
    // We do a cumsum of the zeros along the dimension.
    // For a vector is_zero = [False, True, False, True, False]
    // we would have cumsum = [0, 1, 1, 2, 2]
    // As such we have (in python code for simplicity)
    // The mask for the range [0, z1):
    // cumsum == 0
    // The indices of the first zero z1 and zeros when
    // there is no first zero:
    // indices = (cumsum == 1).max(dim, keepdim=True).indices
    // The mask for the first zero:
    // zeros_like(indices).scatter_(dim, indices, 1.) & cumsum == 1
    // Note that the logic_and with cumsum == 1 accounts
    // for the case when there is no first zero
    Tensor grad_input = at::zeros_symint(input.sym_sizes(), grad.options());
    const auto cumsum = is_zero.cumsum(dim);

    // case k < z1
    // select everything before the first zero [0, z1)
    auto mask = cumsum == 0;
    // equiv to grad_input[mask] = deriv[grad]
    grad_input.masked_scatter_(mask,
        reversed_cumsum(w.masked_fill(~mask, 0.), dim).div_(input_conj).masked_select(mask));
    // select everything from the first zero to the second zero [z1, z2)
    mask = cumsum == 1;

    // case k = z1
    // We start by select the first zero [z1]
    // We locate the indices of the first zero using the max function
    // We then go from the indices to a mask index_fill_
    // When there is no zero in the slice, max will return the index 0.
    // To account for this, we need to do an intersection with mask,
    // which is true in the range [z1, z2)
    const auto first_zero_index = std::get<1>(mask.max(dim, /*keepdim*/ true));
    const auto first_zero_mask = at::zeros_like(mask)
                                  .scatter_(dim, first_zero_index, /*src*/ 1)
                                  .logical_and_(mask);

    // select everything between the first zero and the second zero (z1, z2)
    mask &= ~first_zero_mask;
    // here we compute
    // dy_j / dx_z1 = sum(cumprod(input[z1+1:z2] * grad[z1+1:z2])) * prod(output[z1-1])
    // relu_() necessary as gather does not support negative indices
    // finally, we do grad_input[z1] = dy_j / dx_z1
    grad_input.masked_scatter_(first_zero_mask,
                               input_conj.masked_fill(~mask, 1.).cumprod(dim)
                                    .mul_(grad.masked_fill(cumsum != 1, 0.))
                                    .sum(dim, /*keepdim*/true)
                                    .mul_(at::gather(output_conj, dim, (first_zero_index - 1).relu_())
                                          .masked_fill_(first_zero_index == 0, 1.))
                                    .masked_select(first_zero_mask));
    return grad_input;
  } else { // GradMode::enabled()
    /*
    If the input is nonzero, we need to calculate the dy_j / dx_k
    by using the formula (2), called in the code omitted_products.

    The way the code calculates it is simply by noting that

    prod_{1 <= i <= j, i != k} x_i
        = (prod_{1 <= i <= k} x_i) * (prod_{k + 1 <= i <= j} x_i)

    the first term is calculated as prods_until_k, which since
    doesn't depend in j is easy to vectorize.

    The second term (indexed by j) is the cumulative product of
    x_{k+1}, x_{k+2}, ..., x_n, and it's named in the code
    prods_from_k_pkus_1, and it's calculated as a cumprod.

    In order to vectorize this properly, we need to add to
    omitted_products the dimensions where k > j, and therefore
    dy_j / dx_k = 0, which is done right after the assert.
    */

    Tensor grad_input;
    // For Composite Compliance, we will use
    // at::stack on the grad slices, hence the vector.
    std::vector<Tensor> grad_inputs;
    if (are_inputs_tensors_sublcass) {
      grad_inputs.reserve(dim_size);
    } else {
      grad_input = at::zeros(input.sizes(), grad.options());
    }
    auto ones_size = input.sym_sizes().vec();
    ones_size[dim] = 1;
    const Tensor ones = at::ones({1}, grad.options()).expand_symint(ones_size);
    Tensor prods_from_k_plus_1;
    Tensor omitted_products;
    for (const auto k : c10::irange(dim_size)) {
      if (k == 0) {
        prods_from_k_plus_1 = at::cumprod(input_conj.slice(dim, k + 1), dim);
        omitted_products = at::cat({ones, std::move(prods_from_k_plus_1)}, dim);
      } else if (k == dim_size - 1) {
        const Tensor prods_until_k = at::prod(input_conj.slice(dim, 0, k), dim, true);
        omitted_products = prods_until_k;
      } else {
        const Tensor prods_until_k = at::prod(input_conj.slice(dim, 0, k), dim, true);
        prods_from_k_plus_1 = at::cumprod(input_conj.slice(dim, k+1), dim);
        omitted_products = prods_until_k.expand_as(prods_from_k_plus_1) * prods_from_k_plus_1;
        omitted_products = at::cat({prods_until_k, omitted_products}, dim);
      }

      // At this point omitted_products is the same size
      // as input, except on the dimension dim where it's
      // dim_size - k
      TORCH_CHECK(omitted_products.sym_size(dim) == dim_size - k);

      auto grad_slice = at::sum(grad.slice(dim, k) * omitted_products, dim);
      if (are_inputs_tensors_sublcass) {
        grad_inputs.push_back(grad_slice);
      } else {
        grad_input.select(dim, k).copy_(grad_slice);
      }
    }

    return are_inputs_tensors_sublcass ? at::stack(grad_inputs, dim) : std::move(grad_input);
  }
}

// Implement std::is_nan<IntegralType> for MSVC.
namespace {
#ifdef _MSC_VER
template<typename T>
inline std::enable_if_t<std::is_integral_v<T>, bool> isnan_(T x) {
  return false;
}
template<typename T>
inline std::enable_if_t<!std::is_integral_v<T>, bool> isnan_(T x) {
  return std::isnan(x);
}
#else
template<typename T>
inline bool isnan_(T x) {
  return std::isnan(x);
}
#endif
}

template<typename T1, typename T2, typename Operation>
void cummax_cummin_helper(const T1* self_data, T1* values_data, T2* indices_data,
          int self_dim_size, int self_stride, int values_stride, int indices_stride) {
      Operation op;
      T1 out = c10::load(self_data);
      int idx = 0;
      for (const auto i : c10::irange(self_dim_size)) {
        T1 curr_elem = c10::load(&self_data[i*self_stride]);
        if(isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))) {
            out = curr_elem;
            idx = i;
        }
        values_data[i*values_stride] = out;
        indices_data[i*indices_stride] = idx;
      }
}

void cummax_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf,
    self.scalar_type(), "cummax_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::greater_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummax_out(const Tensor& self, int64_t dim, Tensor& values, Tensor& indices) {
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    NoNamesGuard guard;
    at::native::resize_output(values, self.sizes());
    at::native::resize_output(indices, self.sizes());
    if(self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummax_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummax(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummax_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}

void cummin_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf,
    self.scalar_type(), "cummin_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::less_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummin_out(const Tensor& self, int64_t dim, Tensor& values, Tensor& indices) {
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    NoNamesGuard guard;
    at::native::resize_output(values, self.sizes());
    at::native::resize_output(indices, self.sizes());
    if(self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummin_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummin(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummin_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}

Tensor cummaxmin_backward(const Tensor& grad, const Tensor& input, const Tensor& indices, int64_t dim) {
  if (input.sym_numel() == 0) {
    return input;
  }
  auto result = at::zeros_symint(input.sym_sizes(), input.options());

  // for composite compliance, use out-of-place variant of
  // `scatter_add` if `indices` or `grad` is a Tensor Subclass.
  if (areAnyTensorSubclassLike({indices, grad})) {
    return result.scatter_add(dim, indices, grad);
  }
  return result.scatter_add_(dim, indices, grad);
}

static Tensor prepend_append_on_dim(const Tensor& self, const std::optional<Tensor>& prepend, const std::optional<Tensor>& append, int64_t dim) {
  // Helper for diff that handles prepending and appending when at least one is present
  TORCH_INTERNAL_ASSERT(prepend.has_value() || append.has_value(), "either prepend or append must be have value");
  if (!prepend.has_value() && append.has_value()) {
    return at::cat({self, append.value()}, dim);
  } else if (prepend.has_value() && !append.has_value()) {
    return at::cat({prepend.value(), self}, dim);
  } else {
    return at::cat({prepend.value(), self, append.value()}, dim);
  }
}

static inline void diff_check_compatible_shape(const Tensor& self, const std::optional<Tensor>&other, int64_t dim) {
  // Helper for diff that checks whether the shape of the tensor to prepend or append
  // is compatible with that of input
  if (other.has_value()) {
    int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim(), false);

    TORCH_CHECK(
        other.value().dim() == self.dim(),
        "diff expects prepend or append to be the same dimension as input");

    for (const auto i : c10::irange(other.value().dim())) {
      if (i == wrapped_dim) {
        continue;
      }
      TORCH_SYM_CHECK(
          other.value().sym_size(i).sym_eq(self.sym_size(i)),
          "diff expects the shape of tensor to prepend or append to match that of"
          " input except along the differencing dimension;"
          " input.size(", i, ") = ", self.sym_size(i), ", but got"
          " tensor.size(", i, ") = ", other.value().sym_size(i));
    }
  }
}

static inline void diff_check(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>&prepend, const std::optional<Tensor>& append) {
  // Helper for diff that checks whether its parameters are valid
  TORCH_CHECK(
      self.dim() >= 1,
      "diff expects input to be at least one-dimensional");

  TORCH_CHECK(
      n >= 0,
      "order must be non-negative but got ", n);

  diff_check_compatible_shape(self, prepend, dim);
  diff_check_compatible_shape(self, append, dim);
}

static inline Tensor diff_helper(const Tensor& self, int64_t n, int64_t dim) {
  if (n == 0) {
    auto result = at::zeros_like(self);
    result.copy_(self);
    return result;
  }

  auto out_len = self.sym_size(dim) - 1;
  auto result = self;
  bool is_kBool = (self.dtype() == at::kBool);
  n = n > self.sym_size(dim) ? self.sym_size(dim).guard_int(__FILE__, __LINE__) : n;

  for ([[maybe_unused]] const auto i : c10::irange(n)) {
    if (is_kBool) {
      result = at::logical_xor(
        at::narrow_symint(result, dim, 1, out_len),
        at::narrow_symint(result, dim, 0, out_len)
      );
    } else {
      result = at::narrow_symint(result, dim, 1, out_len) - at::narrow_symint(result, dim, 0, out_len);
    }
    out_len = out_len - 1;
  }

  return result;
}

Tensor diff(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>& prepend, const std::optional<Tensor>& append) {
  diff_check(self, n, dim, prepend, append);
  if ((!prepend.has_value() && !append.has_value()) || n == 0) {
    return diff_helper(self, n, dim);
  } else {
    auto a = prepend_append_on_dim(self, prepend, append, dim);
    return diff_helper(a, n, dim);
  }
}

static inline Tensor& diff_out_helper(const Tensor& self, int64_t n, int64_t dim, Tensor& result) {
  if (n == 0) {
    if (resize_output_check_symint(result, self.sym_sizes())) {
      result.resize__symint(self.sym_sizes());
    }
    check_scalar_type_device_layout_equal(result, self);
    return result.copy_(self);
  }

  n = n > self.sym_size(dim) ? self.sym_size(dim).guard_int(__FILE__, __LINE__) : n;
  const auto out_len = self.sym_size(dim) - n;
  auto prev_result = self;

  if (n > 1) {
    prev_result = diff_helper(self, n - 1, dim);
  }

  if (self.dtype() == at::kBool) {
    at::logical_xor_out(
      result,
      at::narrow_symint(prev_result, dim, 1, out_len),
      at::narrow_symint(prev_result, dim, 0, out_len)
    );
  } else {
    at::sub_out(
      result,
      at::narrow_symint(prev_result, dim, 1, out_len),
      at::narrow_symint(prev_result, dim, 0, out_len)
    );
  }

  return result;
}

Tensor& diff_out(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>& prepend, const std::optional<Tensor>& append, Tensor& result) {
  diff_check(self, n, dim, prepend, append);
  if ((!prepend.has_value() && !append.has_value()) || n == 0) {
    return diff_out_helper(self, n, dim, result);
  } else {
    auto a = prepend_append_on_dim(self, prepend, append, dim);
    return diff_out_helper(a, n, dim, result);
  }
}

static void pre_check_gradient(const Tensor& self, std::optional<int64_t> spacing_size, at::OptionalIntArrayRef dim,  int64_t edge_order) {
  // Helper for gradient function to make sure input data satisfies prerequisites
  TORCH_CHECK(self.scalar_type() != ScalarType::Byte, "torch.gradient does not support uint8 input.");
  if (spacing_size.has_value() && !dim.has_value()) {
    // NOTE: If spacing was given as a scalar, the callers of this function
    // create a spacing vector of the expected size, and this check passes
    TORCH_CHECK(spacing_size.value() == self.dim(),
      "torch.gradient expected spacing to be unspecified, a scalar, or a list ",
      "of length equal to 'self.dim() = ", self.dim(), "', since dim argument ",
      "was not given, but got a list of length ", spacing_size.value());
  }
  if (spacing_size.has_value() && dim.has_value()) {
    TORCH_CHECK(spacing_size.value() == static_cast<int64_t>(dim.value().size()),
    "torch.gradient expected spacing to be unspecified, a scalar or it's spacing and dim arguments to have the same length, but got a spacing argument of length ", spacing_size.value(), " and a dim argument of length ", dim.value().size(), "." );
  }
  TORCH_CHECK(edge_order == 1 || edge_order == 2, "torch.gradient only supports edge_order=1 and edge_order=2.");
  if (dim.has_value()) {
    // The following function get called to check whether dim argument satisfies prerequisites.
    // The output of the function is not used for the computation of gradient.
    dim_list_to_bitset(dim.value(), self.dim());
    for (const auto i : c10::irange(dim.value().size())) {
      TORCH_CHECK(self.size(dim.value()[i]) >= edge_order + 1, "torch.gradient expected each dimension size to be at least edge_order+1");
    }
  } else {
    for (const auto i : c10::irange(self.dim())) {
      TORCH_CHECK(self.size(i) >= edge_order + 1, "torch.gradient expected each dimension size to be at least edge_order+1");
    }
  }
}

static std::vector<Tensor> gradient_helper(const Tensor& self, TensorList coordinates, IntArrayRef dim, int64_t edge_order) {
  for (const auto i : c10::irange(coordinates.size())) {
    TORCH_CHECK(self.device() == coordinates[i].device(), "torch.gradient expected each tensor to be on the same device, but got devices ", self.device(), " and ", coordinates[i].device(), "!");
  }

  std::vector<Tensor> result;
  for (const auto i : c10::irange(dim.size())) {
    TORCH_CHECK( coordinates[i].dim() == 1, "torch.gradient expected each element of spacing to have one dimension, but got an element with ", coordinates[i].dim(), " dimensions!");
    int64_t direction = maybe_wrap_dim(dim[i], self.dim());
    Tensor prepend, append;
    std::vector<int64_t> shape(self.dim(),1);
    shape[ direction ] = -1;

    auto ax_dx = coordinates[i].diff(1,0);
    auto dx1 = at::slice(ax_dx, 0, 0, -1);
    auto dx2 = at::slice(ax_dx, 0, 1);
    auto a = (   -dx2    / (dx1*(dx1+dx2)) ).reshape(shape);
    auto b = ( (dx2-dx1) / (dx1*dx2)       ).reshape(shape);
    auto c = (    dx1    / (dx2*(dx1+dx2)) ).reshape(shape);

    auto center = a * at::slice(self, direction, 0, -2) + b * at::slice(self, direction , 1, -1) + c * at::slice(self, direction, 2);
    if (edge_order == 1) {
     prepend = (at::slice(self, direction, 1, 2  ) - at::slice(self, direction, 0, 1   )) / ax_dx[0]  ;
     append  = (at::slice(self, direction, -1    ) - at::slice(self, direction, -2, -1 )) / ax_dx[-1] ;
    } else if (edge_order == 2) {
     a =-(2.0 * ax_dx[0] + ax_dx[1]) / (ax_dx[0] * (ax_dx[0] + ax_dx[1])) ;
     b = (      ax_dx[0] + ax_dx[1]) / (ax_dx[0] * ax_dx[1])       ;
     c = (     -ax_dx[0]           ) / (ax_dx[1] * (ax_dx[0] + ax_dx[1]));
     prepend = a * at::slice(self, direction, 0, 1) + b * at::slice(self, direction, 1, 2) + c * at::slice(self, direction, 2, 3);

     a = (    ax_dx[-1]            ) / (ax_dx[-2] * (ax_dx[-1] + ax_dx[-2]));
     b =-(    ax_dx[-1] + ax_dx[-2]) / (ax_dx[-1] * ax_dx[-2]);
     c = (2 * ax_dx[-1] + ax_dx[-2]) / (ax_dx[-1] * (ax_dx[-1] + ax_dx[-2]));
     append = a * at::slice(self, direction, -3, -2) + b * at::slice(self, direction, -2, -1) + c * at::slice(self, direction, -1);
    }

    result.emplace_back(prepend_append_on_dim(center, prepend, append, direction));
  }
  return result;
}

static std::vector<Tensor> gradient_helper_float(const Tensor& self, ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order) {
  std::vector<Tensor> result;
  for (const auto i : c10::irange(dim.size())) {
      int64_t direction = maybe_wrap_dim(dim[i], self.dim());
      const auto& ax_dx = spacing[i];
      Tensor prepend, append;
      auto center  = (at::slice(self,direction, 2   ) - at::slice(self, direction, 0, -2 ) ) / ax_dx;
      if (edge_order==1) {
        prepend = (at::slice(self,direction, 1, 2) - at::slice(self, direction, 0, 1  ) ) / ax_dx;
        append  = (at::slice(self,direction, -1  ) - at::slice(self, direction, -2, -1) ) / ax_dx ;
      } else if (edge_order==2) {
        prepend = (-1.5 * at::slice(self, direction, 0, 1) + 2 * at::slice(self, direction, 1, 2)   - 0.5 * at::slice(self, direction, 2, 3))/ ax_dx;
        append = (0.5 * at::slice(self, direction, -3, -2) - 2 * at::slice(self, direction, -2, -1) + 1.5 * at::slice(self, direction, -1))  / ax_dx;
      }

      result.emplace_back(prepend_append_on_dim(center/2, prepend, append, direction));
  }
  return result;
}

static std::vector<int64_t> gradient_dim_preprocess(const Tensor& self, std::optional<int64_t> dim) {
  // if gradient dim is provided as an integer, then we need to compute gradient only on this direction.
  // Moreover, if it's not provided at all, then we are interested in gradient for all directions.
  // Finally, if dim is provided as vector of ints, then it is not expected to be called by this function.
  if (dim.has_value()) {
    return std::vector<int64_t>{dim.value()};
  }

  std::vector<int64_t> axis(self.dim());
  std::iota(axis.begin(), axis.end(), 0);
  return axis;
}

std::vector<Tensor> gradient(const Tensor& self, TensorList coordinates, IntArrayRef dim, int64_t edge_order) {
    pre_check_gradient(self,
                       std::optional<int64_t>(coordinates.size()),
                       at::OptionalIntArrayRef(dim),
                       edge_order);
    return gradient_helper(self, coordinates, dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, TensorList coordinates, std::optional<int64_t> dim, int64_t edge_order) {
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  pre_check_gradient(self,
                     std::optional<int64_t>(coordinates.size()),
                     dim.has_value() ? at::OptionalIntArrayRef(processed_dim) : std::nullopt,
                     edge_order);
  return gradient_helper(self, coordinates, processed_dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, c10::ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order) {
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     at::OptionalIntArrayRef(dim),
                     edge_order);
  return gradient_helper_float(self, spacing, dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, ArrayRef<Scalar> spacing, std::optional<int64_t> dim, int64_t edge_order) {
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     dim.has_value() ? at::OptionalIntArrayRef(processed_dim) : std::nullopt,
                     edge_order);
  return gradient_helper_float(self, spacing, processed_dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, const Scalar& unit_size, IntArrayRef dim, int64_t edge_order) {
  // When spacing is given as scalar, while dim is given as IntArrayRef, scalar value need to
  // be taken as unit size at every given dimension element of - dim.
  std::vector<Scalar> spacing(dim.size(), unit_size);
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     at::OptionalIntArrayRef(dim),
                     edge_order);
  return gradient_helper_float(self, spacing, dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, const std::optional<Scalar>& unit_size, std::optional<int64_t> dim, int64_t edge_order) {
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  // When unit_size not provided, it is always assumed to be equal to 1.
  // When dim has integer value it implies we are looking for gradient in the specific direction, however when
  // it is not provided, it means we are interested to find gradient in all directions.
  std::vector<Scalar> spacing(dim.has_value() ? 1 : self.dim(),
                              unit_size.has_value() ? unit_size.value() : 1.0) ;
  pre_check_gradient(self,
                     unit_size.has_value() ?  std::optional<int64_t>(spacing.size()) : std::nullopt,
                     dim.has_value() ? at::OptionalIntArrayRef(processed_dim) : std::nullopt,
                     edge_order);
  return gradient_helper_float(self, spacing, processed_dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, IntArrayRef dim, int64_t edge_order) {
  std::vector<Scalar> spacing(dim.size(), 1.0) ;
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     at::OptionalIntArrayRef(dim),
                     edge_order);
  return gradient_helper_float(self, spacing, dim, edge_order);
}

// ALL REDUCE #################################################################

inline bool should_use_acc_buffer(at::TensorIterator& iter) {
  const auto ndim = iter.ndim();
  if (!iter.device().is_cpu() || iter.noutputs() != 1) {
    return false;
  }
  if (!at::isReducedFloatingType(iter.common_dtype())) {
    return false;
  }
  if (ndim < 2) {
    return false;
  }
  auto out_strides = iter.strides(0);
  for (const auto dim : c10::irange(0, 2)) {
      if (out_strides[dim] != 0) {
        return false;
      }
  }
  return true;
}

TORCH_IMPL_FUNC(sum_out)
(const Tensor& self,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> opt_dtype,
 const Tensor& result) {
  auto iter = meta::make_reduction_from_out_ty(self, result, opt_dim, keepdim, result.scalar_type());
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    // Here is a limitation of TensorIterator reductions for permuted input with lower precision on CPU.
    // Consider the case: TensorIterator coalesces such input and output to >= 2 dims tensors,
    // and the output stride is [0, 0, x, x, ...] with x >= 0 (two reduced dimensions and non-reduced dims).
    // Since the reduction loop only operates on two dimensions at a time,
    // the intermediate sums is forced to do accumulation in the second reduced dim with lower precision.
    // See https://github.com/pytorch/pytorch/issues/83149
    if (should_use_acc_buffer(iter)) {
      auto tmp_output = at::empty(result.sizes(), result.options().dtype(kFloat));
      at::sum_outf(self.to(ScalarType::Float), opt_dim, keepdim, /*dtype=*/std::nullopt, tmp_output);
      result.copy_(tmp_output);
    } else{
      sum_stub(iter.device_type(), iter);
    }
  }
}

Tensor sum(const Tensor &self, std::optional<ScalarType> dtype) {
  return at::sum(self, IntArrayRef{}, false, dtype);
}

Tensor sum(const Tensor& self, DimnameList dim, bool keepdim, std::optional<ScalarType> dtype) {
  return at::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& sum_out(const Tensor& self, DimnameList dim,
                bool keepdim, std::optional<ScalarType> opt_dtype, Tensor& result) {
  return at::sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

Tensor& nansum_out(const Tensor& self, at::OptionalIntArrayRef dim,
                       bool keepdim, std::optional<ScalarType> opt_dtype, Tensor& result) {
  if (self.device().is_cpu()) {
    TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "nansum does not support complex inputs");
  }

  // For integral types, use existing sum as
  // integral types don't have `Nan`.
  if (c10::isIntegralType(self.scalar_type(), true)){
    return at::sum_out(result, self, dim, keepdim, opt_dtype);
  }

  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  auto iter = make_reduction("nansum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result = result.zero_();
  } else {
    nansum_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor nansum(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  return at::native::nansum_out(self, dim, keepdim, dtype, result);
}

namespace {
template<typename scalar_t, typename accscalar_t = at::acc_type<scalar_t, false>>
void inline set_result(Tensor& result, accscalar_t sum)
{
    if constexpr (std::is_integral_v<accscalar_t>) {
      // all integer types get promoted to kLong
      *result.data_ptr<int64_t>() = sum;
    } else {
      *result.data_ptr<scalar_t>() = sum;
    }
}
}
// NOTE: this could be implemented via diag and sum, but this has perf problems,
// see https://github.com/pytorch/pytorch/pull/47305,
Tensor trace_cpu(const Tensor& self) {
  Tensor result;
  // Returns the ScalarType of the self tensor if the tensor is non integral type
  // In the case, self is an integer type tensor, at::kLong is return since promote_integers
  // is set to true
  ScalarType dtype = get_dtype_from_self(self, std::nullopt, true);
  result = at::empty({}, self.options().dtype(dtype));
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "trace", [&] {
    using accscalar_t = at::acc_type<scalar_t, false>;
    accscalar_t sum = 0;
    const auto* t_data = self.const_data_ptr<scalar_t>();

    int64_t t_stride_0, t_stride_1, t_diag_size;

    TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());

    t_stride_0 = self.stride(0);
    t_stride_1 = self.stride(1);

    t_diag_size = std::min(self.size(0), self.size(1));
    for (const auto i : c10::irange(t_diag_size)) {
      sum += t_data[i * (t_stride_0 + t_stride_1)];
    }
    set_result<scalar_t>(result, sum);

  });

  return result;
}

static void impl_func_prod(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    std::optional<ScalarType> dtype,
    const Tensor& result) {
  auto iter = meta::make_reduction_from_out_ty(self, result, dims, keepdim, result.scalar_type());
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    prod_stub(iter.device_type(), iter);
  }
}

TORCH_IMPL_FUNC(prod_out)
(const Tensor& self,
 int64_t dim,
 bool keepdim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  impl_func_prod(self, dim, keepdim, dtype, result);
}

Tensor prod(const Tensor &self, std::optional<ScalarType> opt_dtype) {
  auto dtype = get_dtype_from_self(self, opt_dtype, true);
  auto shape = meta::get_reduction_shape(self, {}, false);
  Tensor result = at::empty(shape, self.options().dtype(dtype));
  impl_func_prod(self, {}, false, dtype, result);
  return result;
}

Tensor prod(const Tensor& self, Dimname dim, bool keepdim, std::optional<ScalarType> dtype) {
  return at::prod(self, dimname_to_position(self, dim), keepdim, dtype);
}

Tensor& prod_out(const Tensor& self, Dimname dim,
                 bool keepdim, std::optional<ScalarType> opt_dtype, Tensor& result) {
  return at::prod_out(result, self, dimname_to_position(self, dim), keepdim, opt_dtype);
}

TORCH_IMPL_FUNC(mean_out)
(const Tensor& self,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> opt_dtype,
 const Tensor& result) {
  ScalarType dtype = result.scalar_type();
  // TODO: the TensorIterator reduction implementation of mean
  // (mean_kernel_impl()) is unvectorized and leads to very poor performance
  // for production workloads. Once that's fixed, the following code can be used
  // in lieu of the sum + divide implementation below.
  if (self.device().is_cpu()) {
    int64_t dim_prod = 1;
    if (!opt_dim.has_value() || opt_dim.value().empty() || self.ndimension() == 0) {
      dim_prod = self.numel();
    } else {
      auto dim = opt_dim.value();
      for (auto d : dim) {
        dim_prod *= self.size(d);
      }
    }
    // For accuracy reasons, BF16/FP16 mean should be computed via the
    // following approach:
    //  cast_fp32 -> sum -> div -> cast_bf16_or_fp16
    //
    // Such an approach is necessary because if we were to choose the same
    // approach for BF16/FP16 as FP32 here, then it would have resulted in
    // the following code-flow -
    // cast_fp32 -> sum -> cast_bf16 -> cast_fp32 -> div -> cast_bf16,
    // which, in turn, does not produce as accurate results.
    bool is_half_type = (dtype == kHalf || dtype == kBFloat16);
    auto sum_out_dtype = is_half_type ? ScalarType::Float : dtype;
    auto result_temp = is_half_type ? result.to(sum_out_dtype) : result;
    // If dtype is FP16 or BF16, self (input tensor) will initially be cast to
    // FP32 in sum_out. This results in having to read that FP32 tensor again,
    // but maybe in the future, we could revise the implementation to not
    // materialize that intermediate FP32 tensor. That approach would probably
    // require some modifications in binary_kernel_reduce_vec(),
    // TensorIteratorBase::for_each(), and
    // TensorIteratorBase::serial_for_each(), apart from sum kernel for CPU.
    at::sum_out(result_temp, self, opt_dim, keepdim, sum_out_dtype).div_(dim_prod);
    // After sum & div, cast result_temp back to BF16 or FP16, if required.
    // It cannot be avoided copy_() if we promotion the out of sum op, because of
    // the result needs to be update and the storage of result tensor cannot be reused
    // by sum op. We do not need explicit call to(dtype) func as copy_() do it.
    if (is_half_type) {
      result.copy_(result_temp);
    }
  } else {
    // device is not CPU
    auto iter = at::meta::make_reduction_from_out_ty(
        self, result, opt_dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      mean_stub(iter.device_type(), iter);
    }
  }
}

Tensor mean(const Tensor &self, std::optional<ScalarType> dtype) {
  return at::mean(self, IntArrayRef{}, false, dtype);
}

Tensor mean(const Tensor& self, DimnameList dim, bool keepdim, std::optional<ScalarType> dtype) {
  return at::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& mean_out(const Tensor& self, DimnameList dim,
                 bool keepdim, std::optional<ScalarType> opt_dtype, Tensor& result) {
  return at::mean_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

Tensor& mean_dtype_out(const Tensor &self, std::optional<ScalarType> dtype, Tensor& result) {
  TORCH_CHECK(
    canCast(self.scalar_type(), result.scalar_type()),
      "mean.dtype_out(): input types can't be cast to the desired output type ",
      result.scalar_type());
  // at::mean_out should make sure dtype and result.scalar_type() are the same
  return at::mean_out(result, self, IntArrayRef{}, false, dtype);
}

// TODO(@heitorschueroff) implement custom kernels for nanmean
Tensor& nanmean_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<ScalarType> opt_dtype,
    Tensor& result) {
  // Check if dtype is an integral type or Bool and raise an error
  TORCH_CHECK(
    !at::isIntegralType(self.scalar_type(), /*includeBool=*/true),
    "nanmean(): integral types and 'Bool' are not supported for nanmean, even for empty tensors.");
  TORCH_CHECK(
      self.is_floating_point() || self.is_complex(),
      "nanmean(): expected input to have floating point or complex dtype but got ",
      self.scalar_type());
  const auto factor = at::native::isnan(self).logical_not_().sum(dim, keepdim);
  at::native::nansum_out(self, dim, keepdim, opt_dtype, result).div_(factor);
  return result;
}

Tensor nanmean(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<ScalarType> opt_dtype) {
  TORCH_CHECK(
      self.is_floating_point() || self.is_complex(),
      "nanmean(): expected input to have floating point or complex dtype but got ",
      self.scalar_type());
  const auto factor =
      at::native::isnan(self.detach()).logical_not_().sum(dim, keepdim);
  return at::nansum(self, dim, keepdim, opt_dtype).div(factor);
}

static Tensor& logsumexp_out_impl(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  // can't take max of empty tensor
  if (self.numel() != 0) {
    // For complex numbers, use the real part to calculate the max. Based on
    // https://scicomp.stackexchange.com/questions/34273/log-sum-exp-trick-for-signed-complex-numbers
    auto maxes = at::amax(at::real(self), dims, true);
    auto maxes_squeezed = (keepdim ? maxes : at::squeeze(maxes, dims));
    maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
    at::sum_out(result, (self - maxes).exp_(), dims, keepdim);
    result.log_().add_(maxes_squeezed);
  } else {
    at::sum_out(result, at::exp(self), dims, keepdim);
    result.log_();
  }
  return result;
}

Tensor& logsumexp_out(const Tensor& self, IntArrayRef dims, bool keepdim, Tensor& result) {
  // Complex type implies floating point type
  TORCH_CHECK(at::isFloatingType(result.scalar_type()) || at::isComplexType(result.scalar_type()),
              "logsumexp(): Expected floating point type for result tensor, but got: ",
              result.scalar_type());
  {
    NoNamesGuard guard;
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
      // for integral inputs, promote input to default floating type.
      auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
      logsumexp_out_impl(result, self.to(default_dtype), dims, keepdim);
    } else {
      logsumexp_out_impl(result, self, dims, keepdim);
    }
  }
  namedinference::propagate_names_for_reduction(result, self, dims, keepdim);
  return result;
}

Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  TensorOptions result_options;
  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // even for integral inputs, result is floating dtype
    auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
    result_options = self.options().dtype(default_dtype);
  } else {
    result_options = self.options();
  }
  auto result = at::empty({0}, result_options);
  return at::logsumexp_outf(self, dims, keepdim, result);
}

Tensor logsumexp(const Tensor& self, DimnameList dims, bool keepdim) {
  return at::logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

Tensor& logsumexp_out(const Tensor& self, DimnameList dims, bool keepdim, Tensor& result) {
  return at::logsumexp_out(result, self, dimnames_to_positions(self, dims), keepdim);
}

// special_logsumexp, alias for logsumexp
Tensor special_logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  return self.logsumexp(dims, keepdim);
}
Tensor& special_logsumexp_out(const Tensor& self, IntArrayRef dims, bool keepdim, Tensor& result) {
  return at::logsumexp_out(result, self, dims, keepdim);
}

static void impl_func_norm(
    const Tensor& self,
    const OptionalScalarRef& opt_p,
    IntArrayRef dim,
    bool keepdim,
    std::optional<ScalarType> opt_dtype,
    const Tensor& result) {
  // Left this implementation without deprecating it as it is called in a number of places
  // in the codebase. We should swap those by linalg_vector_norm
  auto p = opt_p.has_value() ? opt_p.get() : Scalar(2.0).to<double>();
  at::linalg_vector_norm_out(const_cast<Tensor&>(result), self, p, dim, keepdim, opt_dtype);
}

TORCH_IMPL_FUNC(norm_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, std::nullopt, result);
}

TORCH_IMPL_FUNC(norm_dtype_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 ScalarType dtype,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, dtype, result);
}

Tensor sparse_norm(
    const Tensor& self,
    const std::optional<Scalar>& p,
    IntArrayRef dim,
    bool keepdim) {
  return at::native_norm(self, p, dim, keepdim, std::nullopt);
}

Tensor sparse_dtype_norm(
    const Tensor& self,
    const std::optional<Scalar>& p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  return at::native_norm(self, p, dim, keepdim, dtype);
}

Tensor norm(const Tensor& self, const std::optional<Scalar>& p, ScalarType dtype) {
  return at::norm(self, p, IntArrayRef{}, false, dtype);
}

Tensor norm(const Tensor& self, const Scalar& p) {
  return at::norm(self, p, IntArrayRef{}, false);
}

inline TensorIterator get_allany_iter(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef dims,
    bool keepdim) {
  if (self.is_cuda()) {
    // As CUDA supports dynamic type casting, we use this overload of
    // `make_reduction`, which doesn't cast input to the result type i.e. kBool.,
    // otherwise we use the overload below which casts the input to kBool (which is
    // an extra operation).
    return meta::make_reduction(self, result, dims, keepdim, self.scalar_type());
  }
  return meta::make_reduction_from_out_ty(
      self, result, dims, keepdim, result.scalar_type());
}

template <int identity, typename Stub>
inline void allany_impl(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef dims,
    bool keepdim,
    Stub& stub) {
  if (self.numel() == 0) {
    result.fill_(identity);
  } else if (self.numel() == 1) {
    result.copy_(self.view_as(result).to(at::kBool));
  } else {
    auto iter = get_allany_iter(self, result, dims, keepdim);
    stub(iter.device_type(), iter);
  }
}

TORCH_IMPL_FUNC(all_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  allany_impl<1>(self, result, dim, keepdim, and_stub);
}

TORCH_IMPL_FUNC(all_dims_out)
(const Tensor& self, OptionalIntArrayRef dim, bool keepdim, const Tensor& result) {
  allany_impl<1>(self, result, dim, keepdim, and_stub);
}

TORCH_IMPL_FUNC(all_all_out)(const Tensor& self, const Tensor& result) {
  allany_impl<1>(self, result, {}, false, and_stub);
}

TORCH_IMPL_FUNC(any_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  allany_impl<0>(self, result, dim, keepdim, or_stub);
}

TORCH_IMPL_FUNC(any_dims_out)
(const Tensor& self, OptionalIntArrayRef dim, bool keepdim, const Tensor& result) {
  allany_impl<0>(self, result, dim, keepdim, or_stub);
}

TORCH_IMPL_FUNC(any_all_out)(const Tensor& self, const Tensor& result) {
  allany_impl<0>(self, result, {}, false, or_stub);
}

template <bool is_all>
Tensor allany_dims_default(const Tensor &self, OptionalIntArrayRef dim, bool keepdim) {
  // Default implementation in terms of all-reduce or single dim reduce
  if (!dim) {
    Tensor out;
    if constexpr (is_all) {
      out = self.all();
    } else {
      out = self.any();
    }

    if (keepdim) {
      DimVector out_shape(self.dim(), 1);
      return out.expand(out_shape);
    }
    return out;
  }

  if (dim->empty()) {
    if (self.scalar_type() == kByte) {
      // Convert to a 1 or 0 mask
      auto out = at::empty_like(self);
      return at::ne_outf(self, 0, out);
    } else {
      return at::_to_copy(self, kBool);
    }
  }

  Tensor out = self;
  for (auto d : *dim) {
    if constexpr (is_all) {
      out = out.all(d, /*keepdim=*/true);
    } else {
      out = out.any(d, /*keepdim=*/true);
    }
  }
  return keepdim ? out : out.squeeze(*dim);
}

Tensor all_dims_default(const Tensor &self, OptionalIntArrayRef dim, bool keepdim) {
  return allany_dims_default<true>(self, dim, keepdim);
}

Tensor any_dims_default(const Tensor &self, OptionalIntArrayRef dim, bool keepdim) {
  return allany_dims_default<false>(self, dim, keepdim);
}

Tensor& all_dims_out_default(
    const Tensor &self, OptionalIntArrayRef dim, bool keepdim, Tensor &result) {
  TORCH_CHECK(self.device() == result.device(), "all.dims: output must be on the same device as input");
  auto tmp = all_dims_default(self, dim, keepdim);
  at::native::resize_output(result, tmp.sizes());
  return result.copy_(tmp);
}

Tensor& any_dims_out_default(
    const Tensor &self, OptionalIntArrayRef dim, bool keepdim, Tensor &result) {
  TORCH_CHECK(self.device() == result.device(), "any.dims: output must be on the same device as input");
  auto tmp = any_dims_default(self, dim, keepdim);
  at::native::resize_output(result, tmp.sizes());
  return result.copy_(tmp);
}

TORCH_IMPL_FUNC(amin_out) (const Tensor& self, IntArrayRef dim, bool keepdim, const Tensor& result) {
  auto iter =
      meta::make_reduction(self, result, dim, keepdim, self.scalar_type());
  if (iter.numel() != 0) {
    min_values_stub(iter.device_type(), iter);
  }
}

TORCH_IMPL_FUNC(amax_out) (const Tensor& self, IntArrayRef dim, bool keepdim, const Tensor& result) {
  auto iter =
      meta::make_reduction(self, result, dim, keepdim, self.scalar_type());
  if (iter.numel() != 0) {
    max_values_stub(iter.device_type(), iter);
  }
}

template <class Stub>
void argmax_argmin_impl(
    const Tensor& self,
    std::optional<int64_t> dim,
    bool keepdim,
    const Tensor& result,
    Stub& stub) {
  c10::MaybeOwned<Tensor> in;
  DimVector dims;
  int64_t _dim = 0;

  if (dim.has_value()) {
    _dim = maybe_wrap_dim(dim.value(), self.dim());
    auto sizes = self.sizes();

    if (sizes[_dim] == 1) {
      result.fill_(0);
      return;
    }

    dims = IntArrayRef(_dim);
    in = c10::MaybeOwned<Tensor>::borrowed(self);
  } else {
    in = c10::MaybeOwned<Tensor>::owned(self.reshape({-1}));
    keepdim = false;
  }

  auto iter =
      meta::make_reduction(*in, result, dims, keepdim, self.scalar_type());

  if (iter.numel() != 0) {
    stub(iter.device_type(), iter);
  }
}

TORCH_IMPL_FUNC(argmax_out)
(const Tensor& self,
 std::optional<int64_t> dim,
 bool keepdim,
 const Tensor& result) {
  argmax_argmin_impl(self, dim, keepdim, result, argmax_stub);
}

TORCH_IMPL_FUNC(argmin_out)
(const Tensor& self,
 std::optional<int64_t> dim,
 bool keepdim,
 const Tensor& result) {
  argmax_argmin_impl(self, dim, keepdim, result, argmin_stub);
}

static double std_var_all_cpu(const Tensor& self, double correction, bool take_sqrt) {
  const auto dtype = self.scalar_type();
  TORCH_CHECK(dtype == kDouble || dtype == kFloat,
              "std_var_all: Unsupported dtype ", dtype);

  auto mean = self.mean().item<double>();
  auto iter = TensorIteratorConfig()
      .add_const_input(self)
      .build();

  auto reduction = [&](int64_t begin, int64_t end, double thread_sum) {
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "std_var_all_cpu", [&] {
      iter.serial_for_each([&] (char** data, const int64_t* strides, int64_t size0, int64_t size1) {
        const double local_mean = mean;
        const int64_t inner_stride = strides[0];
        const int64_t outer_stride = strides[1];

        double local_sum = 0.0;
        for (const auto i : c10::irange(size1)) {
          const char* row_ptr = data[0] + outer_stride * i;
          for (const auto j : c10::irange(size0)) {
            const auto ptr = reinterpret_cast<const scalar_t*>(row_ptr + inner_stride * j);
            auto dx = (static_cast<double>(*ptr) - local_mean);
            local_sum += dx * dx;
          }
        }
        thread_sum += local_sum;
      }, {begin, end});
    });

    return thread_sum;
  };

  // ((x - mean)**2).sum()
  const double sum_dx2 = at::parallel_reduce(
      0, iter.numel(), at::internal::GRAIN_SIZE, 0.0, reduction, std::plus<>{});

  const auto var = [&] () __ubsan_ignore_float_divide_by_zero__ {
    return sum_dx2 / std::max(0.0, self.numel() - correction);
  }();
  const auto result = take_sqrt ? std::sqrt(var) : var;

  if (dtype == kFloat) {
    // Convert to infinity if out of range for a float.
    // Doing it now prevents checked_convert failing later
    return static_cast<float>(result);
  }
  return result;
}

namespace {
  inline void warn_invalid_degrees_of_freedom(const char* fname, const TensorIterator& iter, double correction) {
    int64_t reducing_over_num_elements = iter.num_output_elements() == 0 ? 0 : iter.numel() / iter.num_output_elements();
    if (reducing_over_num_elements - correction <= 0) {
      TORCH_WARN(fname, "(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel).");
    }
  }
} // namespace

static Tensor& std_var_out(
    const char* fname, Tensor& result, const Tensor& self,
    at::OptionalIntArrayRef dim, const std::optional<Scalar>& correction_opt,
    bool keepdim, bool take_sqrt) {
  TORCH_CHECK(self.device().is_cpu() || self.device().is_cuda() || self.device().is_xpu(),
              "std and var supports tensors on a CPU, CUDA, or XPU device only, but got: ",
              self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "std and var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "std and var only support floating point and complex dtypes");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate variance of real and imaginary components
    // separately then add to get overall variance.
    ScalarType dtype = c10::toRealValueType(get_dtype_from_result(result, {}));
    Tensor real_in = at::real(self);
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out(
        fname,
        real_out,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out(
        fname,
        imag_out,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    at::add_out(result, real_out, imag_out);
    if (take_sqrt) {
      at::sqrt_out(result, result);
    }
    return result;
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result, {});
  auto iter = make_reduction(fname, result, self, dim, keepdim, dtype);
  TORCH_CHECK(at::canCast(self.scalar_type(), result.scalar_type()),
              "result type ", self.scalar_type(), " can't be cast to the "
              "desired output type ", result.scalar_type());
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    result.fill_(std::numeric_limits<double>::quiet_NaN());
    return result;
  } else if (
      result.numel() == 1 && iter.device_type() == kCPU &&
      iter.common_dtype() != kBFloat16 && iter.common_dtype() != kHalf) {
    // NOTE: CPU performance significantly regressed when attempting to port to
    // ATen,
    //   so all-reduce has a custom implementation.
    //   See https://github.com/pytorch/pytorch/pull/43858.
    result.fill_(std_var_all_cpu(self, correction, take_sqrt));
  } else {
    std_var_stub(iter.device_type(), iter, correction, take_sqrt);
  }
  return result;
}

static std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname, Tensor& result1, Tensor& result2, const Tensor& self,
    at::OptionalIntArrayRef dim, const std::optional<Scalar>& correction_opt,
    bool keepdim, bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda() || self.is_xpu(),
              fname, " supports tensors on a CPU, CUDA, or XPU device only, got: ",
              self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              fname, " only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              fname, " only support floating point and complex dtypes");
  TORCH_CHECK(result1.scalar_type() == c10::toRealValueType(result2.scalar_type()),
              fname, " expected result1 to be real and match the precision of result2. Got ",
              result1.scalar_type(), " and ", result2.scalar_type(), ".");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate for real and imaginary components separately then combine as:
    // variance = var_real + var_imag
    // mean = mean_real + j * mean_imag
    ScalarType dtype = c10::toRealValueType(get_dtype_from_result(result1, {}));
    Tensor real_in = at::real(self);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        real_out_var,
        real_out_mean,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        imag_out_var,
        imag_out_mean,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    at::add_out(result1, real_out_var, imag_out_var);
    if (take_sqrt) {
      at::sqrt_out(result1, result1);
    }
    at::complex_out(result2, real_out_mean, imag_out_mean);
    return std::tuple<Tensor&, Tensor&>(result1, result2);
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result1, {});
  auto iter =
      make_reduction(fname, result1, result2, self, dim, keepdim, dtype);
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    result1.fill_(std::numeric_limits<double>::quiet_NaN());
    result2.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    std_var_stub(iter.device_type(), iter, correction, take_sqrt);
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

std::tuple<Tensor, Tensor> var_mean(
    const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::var_mean(
      self, /*dim=*/at::OptionalIntArrayRef(dim),
      /*correction=*/Scalar(unbiased ? 1 : 0),
      keepdim);
}

std::tuple<Tensor, Tensor> std_mean(
    const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::std_mean(
      self, /*dim=*/at::OptionalIntArrayRef(dim),
      /*correction=*/Scalar(unbiased ? 1 : 0),
      keepdim);
}

std::tuple<Tensor, Tensor> std_mean(const Tensor& self, bool unbiased) {
  return at::std_mean(
      self, /*dim=*/std::nullopt,
      /*correction=*/Scalar(unbiased ? 1 : 0));
}

std::tuple<Tensor, Tensor> var_mean(const Tensor& self, bool unbiased) {
  return at::var_mean(
      self, /*dim=*/std::nullopt,
      /*correction=*/Scalar(unbiased ? 1 : 0));
}
std::tuple<Tensor&, Tensor&> var_mean_out(
    Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim,
    int64_t correction, bool keepdim) {
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

static TensorOptions options_to_value_type(TensorOptions opts) {
  auto scalar_type = typeMetaToScalarType(opts.dtype());
  return opts.dtype(c10::toRealValueType(scalar_type));
}

std::tuple<Tensor, Tensor> var_mean(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> std_mean(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "std_mean", result1, result2, self, dim, correction, keepdim, true);
}

Tensor var(const Tensor& self, bool unbiased) {
  return at::var(
      self, /*dim=*/std::nullopt,
      /*correction=*/Scalar(unbiased ? 1 : 0));
}

Tensor var(const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::var(
      self, /*dim=*/at::OptionalIntArrayRef(dim),
      /*correction=*/Scalar(unbiased ? 1 : 0),
      keepdim);
}

Tensor& var_out(const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim, Tensor& result) {
  return at::var_out(
      result, self, /*dim=*/at::OptionalIntArrayRef(dim),
      /*correction=*/Scalar(unbiased ? 1 : 0),
      keepdim);
}

Tensor std(const Tensor& self, bool unbiased) {
  return at::std(
      self, /*dim=*/std::nullopt, /*correction=*/Scalar(unbiased ? 1 : 0));
}

Tensor std(const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::std(self, dim,
                 /*correction=*/Scalar(unbiased ? 1 : 0), keepdim);
}

Tensor& std_out(const Tensor& self, at::OptionalIntArrayRef opt_dim, bool unbiased, bool keepdim, Tensor& result) {
  return at::std_out(result, self, opt_dim,
                     /*correction=*/Scalar(unbiased ? 1 : 0), keepdim);
}

Tensor std(const Tensor& self, at::OptionalIntArrayRef dim,
           const std::optional<Scalar>& correction, bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& std_out(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim, Tensor& result) {
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& var_out(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim, Tensor& result) {
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor var(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor std(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& std_out(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim, Tensor& result) {
  return at::std_out(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor var(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& var_out(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim, Tensor& result) {
  return at::var_out(
      result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor std(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction, bool keepdim) {
  return at::std(self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor& std_out(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction,
                bool keepdim, Tensor& result) {
  return at::std_out(result, self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor var(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction, bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor& var_out(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction,
                bool keepdim, Tensor& result) {
  return at::var_out(
      result, self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim,
                                   const std::optional<Scalar>& correction, bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim,
                                   const std::optional<Scalar>& correction, bool keepdim) {
  return at::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor& norm_out(const Tensor& self, const std::optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& norm_out(const Tensor& self, const std::optional<Scalar>& p, DimnameList dim, bool keepdim, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim);
}

Tensor norm(const Tensor& self, const std::optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor norm(const Tensor& self, const std::optional<Scalar>& p, DimnameList dim, bool keepdim) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim);
}

Tensor any(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("any");
}
Tensor& any_out(const Tensor &self, Dimname dim, bool keepdim, Tensor& result) {
  reportNYIDimnameOverload("any");
}
Tensor all(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("all");
}
Tensor& all_out(const Tensor &self, Dimname dim, bool keepdim, Tensor& result) {
  reportNYIDimnameOverload("all");
}
Tensor _is_all_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.all();
}
Tensor _is_any_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.any();
}
Tensor logcumsumexp(const Tensor& self, Dimname dim) {
  return at::logcumsumexp(self, dimname_to_position(self, dim));
}
Tensor& logcumsumexp_out(const Tensor& self, Dimname dim, Tensor& result) {
  return at::logcumsumexp_out(result, self, dimname_to_position(self, dim));
}
Tensor cumsum(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumsum(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumsum_(Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumsum_out(self, self, dimname_to_position(self, dim), dtype);
}
Tensor& cumsum_out(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype, Tensor& result) {
  return at::cumsum_out(result, self, dimname_to_position(self, dim), dtype);
}
Tensor cumprod(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumprod(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumprod_(Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumprod_out(self, self, dimname_to_position(self, dim), dtype);
}
Tensor& cumprod_out(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype, Tensor& result) {
  return at::cumprod_out(result, self, dimname_to_position(self, dim), dtype);
}
std::tuple<Tensor, Tensor> cummax(const Tensor& self, Dimname dim) {
  return at::cummax(self, dimname_to_position(self, dim));
}
std::tuple<Tensor&, Tensor&> cummax_out(const Tensor& self, Dimname dim, Tensor& values, Tensor& indices) {
  return at::cummax_out(values, indices, self, dimname_to_position(self, dim));
}
std::tuple<Tensor, Tensor> cummin(const Tensor& self, Dimname dim) {
  return at::cummin(self, dimname_to_position(self, dim));
}
std::tuple<Tensor&, Tensor&> cummin_out(const Tensor& self, Dimname dim, Tensor& values, Tensor& indices) {
  return at::cummin_out(values, indices, self, dimname_to_position(self, dim));
}

Tensor dist(const Tensor &self, const Tensor& other, const Scalar& p){
  return at::norm(self - other, p);
}

bool cpu_equal(const Tensor& self, const Tensor& other) {
  if (!at::namedinference::are_names_equal(
        self.unsafeGetTensorImpl(), other.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == other.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", other.device());
  if (!self.is_same_size(other)) {
    return false;
  }
  // Since the flags like neg/conj should be already handled outside the
  // TensorIterator, it should be safe to have the following fast path by
  // ensuring the storage and strides exactly the same.
  if (self.is_alias_of(other)
      && self.storage_offset() == other.storage_offset()
      && self.dtype() == other.dtype()
      && self.is_contiguous() == other.is_contiguous()
      && self.strides().equals(other.strides())
      // Extra checks to ensure the safety in case cpu_equal is directly called in C++.
      && self.layout() == other.layout()
      && self.is_neg() == other.is_neg()
      && self.is_conj() == other.is_conj()) {
    if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
      return true;
    }
    std::atomic<bool> result{true};
    auto iter = TensorIteratorConfig().add_const_input(self).build();
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "equal_notnan_cpu", [&] {
      iter.for_each([&](char** data, const int64_t *strides, int64_t dim_size) {
        if (!result) {
            return;
        }
        char* self_data = data[0];
        for ([[maybe_unused]] const auto i : c10::irange(dim_size)) {
          if (isnan_(c10::load<scalar_t>(self_data))) {
            result = false;
            return;
          }
          self_data += strides[0];
        }
      });
    });
    return result.load();
  }

  std::atomic<bool> result{true};
  auto iter = TensorIteratorConfig()
    .add_const_input(self)
    .add_const_input(other)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .build();

  AT_DISPATCH_V2(iter.input_dtype(), "equal_cpu", AT_WRAP([&] {
    iter.for_each([&](char** data, const int64_t *strides, int64_t dim_size) {
      if (!result) {
          return;
      }
      char* self_data = data[0];
      char* other_data = data[1];
      for ([[maybe_unused]] const auto i : c10::irange(dim_size)) {
        if (c10::load<scalar_t>(self_data) != c10::load<scalar_t>(other_data)) {
          result = false;
          return;
        }
        self_data += strides[0];
        other_data += strides[1];
      }
    });
  }), kBool, kBFloat16, kHalf, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return result.load();
}

// max(dim), min(dim), topk(dim), mode(dim), are examples of reduction
// functions that select values. value_selecting_reduction_backward is the
// backward function for those operators; it propagates the grad to the
// specific value locations referred to at `indices`.
Tensor value_selecting_reduction_backward_symint(const Tensor& grad, int64_t dim, const Tensor& indices, c10::SymIntArrayRef sizes, bool keepdim) {
  auto inplace_scatter_if_not_tensor_subclass =
      [&](const Tensor& grad_out, const Tensor& indices_) {
        auto grad_in = at::zeros_symint(sizes, grad_out.options());
        if (areAnyTensorSubclassLike({grad, indices})) {
          return grad_in.scatter(dim, indices_, grad_out);
        }
        return grad_in.scatter_(dim, indices_, grad_out);
      };

  if (!keepdim && !sizes.empty()) {
    auto grad_ = grad.unsqueeze(dim);
    auto indices_ = indices.unsqueeze(dim);
    return inplace_scatter_if_not_tensor_subclass(grad_, indices_);
  }
  return inplace_scatter_if_not_tensor_subclass(grad, indices);
}

Tensor sum_csr(const Tensor &self, std::optional<ScalarType> dtype) {
  return self.values().sum(dtype);
}

Tensor sum_coo(const Tensor &self, std::optional<ScalarType> dtype) {
  return self._values().sum(dtype);
}

Tensor sum_sparse_coo(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<ScalarType> dtype) {
  Tensor result;
  if (dim.has_value()) {
    if (dtype.has_value()) {
      result = at::_sparse_sum(self, *dim, *dtype);
    } else {
      if (c10::isIntegralType(self.scalar_type(), true)) {
        result = at::_sparse_sum(self, *dim, at::kLong);
      } else {
        result = at::_sparse_sum(self, *dim);
      }
    }
  } else {
    result = sum_coo(self, dtype);
  }
  if (keepdim) {
    auto dim_mask = make_dim_mask(dim, self.dim());
    for (int dim = 0; dim < self.dim(); dim++) {
      if (dim_mask[dim]) {
        result = result.unsqueeze(dim);
      }
    }
  }
  return result;
}

Tensor sum_sparse_compressed(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  // TODO: The signature of sum.dim_IntList and _sparse_csr_sum.dim_dtype is a little
  // bit different in the second parameters `dim`, which causes the conversion of `dim`
  // to call into `_sparse_csr_sum`. Align the signatures would be a better choice.
  TORCH_CHECK(
      dim.has_value(), "dim has no value, cannot be used in sum.dim_IntList");
  auto layout = self.layout();
  TORCH_CHECK(
      layout == kSparseCsr,
      "Currently the only compressed sparse format supported for sum.dim_IntList is CSR, but got layout ",
      layout)
  return at::_sparse_csr_sum(self, *dim, keepdim, dtype);
}

} // namespace at::native
