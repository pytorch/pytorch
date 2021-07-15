#include <ATen/native/ReduceOps.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TensorDimApply.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/core/grad_mode.h>

#include <c10/util/irange.h>
#include <c10/util/SmallBuffer.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <map>
#include <cmath>
#include <cfloat>
#include <type_traits>

namespace at {
namespace meta {

ScalarType check_allany_and_get_output_dtype(
    const char* name,
    const Tensor& self,
    const Tensor& result,
    bool keepdim) {
  // Refer [all, any : uint8 compatibility]
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      name, " only supports strided layout, got: ",
      self.layout());

  ScalarType out_dtype;

  if (result.defined()) {
    // Refer [all, any : uint8 compatibility]
    TORCH_CHECK(
        result.scalar_type() == ScalarType::Bool ||
            result.scalar_type() == ScalarType::Byte,
        name, " only supports bool tensor for result, got: ",
        result.scalar_type());
    out_dtype = result.scalar_type();
  } else {
    if (self.scalar_type() == ScalarType::Byte) {
      out_dtype = self.scalar_type();
    } else {
      out_dtype = ScalarType::Bool;
    }
  }

  return out_dtype;
}

void check_allany_for_meta(
    impl::MetaBase& meta,
    const char* name,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  dim = maybe_wrap_dim(dim, self.dim());
  const auto& result = meta.maybe_get_output();
  auto out_dtype = check_allany_and_get_output_dtype(name, self, result, keepdim);
  auto shape = get_reduction_shape(self, dim, keepdim);
  meta.set_output(shape, self.options().dtype(out_dtype));
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
}

TORCH_META_FUNC2(all, dim)(const Tensor& self, int64_t dim, bool keepdim) {
  check_allany_for_meta(*this, "all", self, dim, keepdim);
}

TORCH_META_FUNC2(any, dim)(const Tensor& self, int64_t dim, bool keepdim) {
  check_allany_for_meta(*this, "any", self, dim, keepdim);
}

void check_argmax_argmin(
    impl::MetaBase& meta,
    const char* name,
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  DimVector shape;

  if (dim.has_value()) {
    auto _dim = maybe_wrap_dim(dim.value(), self.dim());
    native::zero_numel_check_dims(self, _dim, name);
    shape = get_reduction_shape(self, _dim, keepdim);
  } else {
    TORCH_CHECK_INDEX(
        self.numel() != 0,
        name, ": Expected reduction dim to be specified for input.numel() == 0.");
  }

  meta.set_output(shape, self.options().dtype(kLong));
}

TORCH_META_FUNC(argmax)
(const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  check_argmax_argmin(*this, "argmax", self, dim, keepdim);
}

TORCH_META_FUNC(argmin)
(const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  check_argmax_argmin(*this, "argmin", self, dim, keepdim);
}

} // namespace meta

namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sum_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(nansum_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(std_var_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(prod_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(norm_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(mean_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(and_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(or_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(min_values_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(max_values_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(argmax_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(argmin_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cumsum_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cumprod_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

Tensor _cumsum_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  cumsum_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor& _cumsum_out_cpu(const Tensor& self, int64_t dim, Tensor& result) {
  cumsum_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor cumsum(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_cumsum(integer_upcast(self, dtype), dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& cumsum_(Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
          !dtype.has_value() || (self.scalar_type() == dtype.value()),
          "provided dtype must match the dtype of self tensor in cumsum. Got ",
          toString(self.scalar_type()),
          " and ",
          toString(dtype.value()),
          ".");

  return at::_cumsum_out(self, self, dim);
}

Tensor& cumsum_out(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype, Tensor& result) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumsum. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  {
    NoNamesGuard guard;
    at::_cumsum_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

Tensor _cumprod_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  cumprod_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor& _cumprod_out_cpu(const Tensor& self, int64_t dim, Tensor& result) {
  cumprod_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor cumprod(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_cumprod(integer_upcast(self, dtype), dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& cumprod_(Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
    TORCH_CHECK(
            !dtype.has_value() || (self.scalar_type() == dtype.value()),
            "provided dtype must match the dtype of self tensor in cumprod. Got ",
            toString(self.scalar_type()),
            " and ",
            toString(dtype.value()),
            ".");

    return at::_cumprod_out(self, self, dim);
}

Tensor& cumprod_out(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype, Tensor& result) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumprod. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  {
    NoNamesGuard guard;
    at::_cumprod_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

Tensor reversed_cumsum(const Tensor& w, int64_t dim) {
  return w.flip(dim).cumsum(dim).flip(dim);
}

Tensor cumprod_backward(const Tensor& grad, const Tensor& input, int64_t dim, const Tensor& output) {
  /*
    We show here how to derive an O(n) gradient formula for
    abitrary inputs. It follows via a basic application of the
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

    The term (dy_j / dx_k) is easilly seen to be

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

    When the imputs are complex, this is map is holomorphic. As such, to compute
    its backwards is just the conjugate of the usual backwards. This simplifies to
    conjugating the input. We may also reuse the output as, since the map is holomorphic,
    cumprod(input.conj()) = cumprod(input).conj()
  */

  if (input.numel() <= 1) {
    return grad;
  }
  dim = at::maybe_wrap_dim(dim, input.dim());
  const int64_t dim_size = input.sizes()[dim];
  if (dim_size == 1) {
    return grad;
  }

  // To enable complex support.
  // From this line on `input_conj` and output_conj`
  // are interchangeable with `input` and `output`.
  auto input_conj = input.conj();
  auto output_conj = output.conj();

  const auto w = output_conj * grad;
  const auto is_zero = input == 0;
  if (!(is_zero.any().item<uint8_t>())) {
    return reversed_cumsum(w, dim).div(input_conj);
  }

  // If we are not computing a second order gradient, we can use an
  // O(n) implementation. The derivative of this implementation is _not_
  // the second derivative of cumprod. As such, we fallback to a less efficient
  // O(n^2) implementation when at::GradMode::is_enabled().
  Tensor grad_input = at::zeros(input.sizes(), grad.options());
  if (!at::GradMode::is_enabled()) {
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

    auto ones_size = input.sizes().vec();
    ones_size[dim] = 1;
    const Tensor ones = at::ones({1}, grad.options()).expand(ones_size);
    Tensor prods_from_k_plus_1;
    Tensor omitted_products;
    for (const auto k : c10::irange(dim_size)) {
      if (k == 0) {
        prods_from_k_plus_1 = at::cumprod(input_conj.slice(dim, k + 1), dim);
        omitted_products = at::cat({ones, prods_from_k_plus_1}, dim);
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
      TORCH_CHECK(omitted_products.size(dim) == dim_size - k);

      grad_input.select(dim, k).copy_(
          at::sum(grad.slice(dim, k) * omitted_products,dim));
    }
  }
  return grad_input;
}

// Implement std::is_nan<IntegralType> for MSVC.
namespace {
#ifdef _MSC_VER
template<typename T>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type isnan_(T x) {
  return false;
}
template<typename T>
inline typename std::enable_if<!std::is_integral<T>::value, bool>::type isnan_(T x) {
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
      T1 out = self_data[0];
      int idx = 0;
      for(int i = 0; i < self_dim_size; i++) {
        T1 curr_elem = self_data[i*self_stride];
        if(isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))) {
            out = self_data[i*self_stride];
            idx = i;
        }
        values_data[i*values_stride] = out;
        indices_data[i*indices_stride] = idx;
      }
}

void cummax_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
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
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
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
  if (input.numel() == 0) {
    return input;
  }
  auto result = at::zeros(input.sizes(), input.options());
  return result.scatter_add_(dim, indices, grad);
}

static Tensor prepend_append_on_dim(const Tensor& self, const c10::optional<Tensor>& prepend, const c10::optional<Tensor>& append, int64_t dim) {
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

static inline void diff_check_compatible_shape(const Tensor& self, const c10::optional<Tensor>&other, int64_t dim) {
  // Helper for diff that checks whether the shape of the tensor to prepend or append
  // is compatible with that of input
  if (other.has_value()) {
    int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim(), false);

    TORCH_CHECK(
        other.value().dim() == self.dim(),
        "diff expects prepend or append to be the same dimension as input");

    for (int i = 0; i < other.value().dim(); i++) {
      TORCH_CHECK(
          other.value().size(i) == self.size(i) || i == wrapped_dim,
          "diff expects the shape of tensor to prepend or append to match that of"
          " input except along the differencing dimension;"
          " input.size(", i, ") = ", self.size(i), ", but got"
          " tensor.size(", i, ") = ", other.value().size(i));
    }
  }
}

static inline void diff_check(const Tensor& self, int64_t n, int64_t dim, const c10::optional<Tensor>&prepend, const c10::optional<Tensor>& append) {
  // Helper for diff that checks whether its parameters are valid
  TORCH_CHECK(
      n == 1,
      "diff only supports n = 1 currently. Please file an issue at"
      " https://github.com/pytorch/pytorch/issues/new?assignees=&labels=&template=feature-request.md"
      " if your use case requires supporting higher-order differences");

  TORCH_CHECK(
      self.dim() >= 1,
      "diff expects input to be at least one-dimensional");

  diff_check_compatible_shape(self, prepend, dim);
  diff_check_compatible_shape(self, append, dim);
}

static inline Tensor diff_helper(const Tensor& self, int64_t n, int64_t dim) {
  auto out_len = self.size(dim) - 1;
  if (self.dtype() == at::kBool) {
    return at::logical_xor(at::narrow(self, dim, 1, out_len), at::narrow(self, dim, 0, out_len));
  }
  return at::narrow(self, dim, 1, out_len) - at::narrow(self, dim, 0, out_len);
}

Tensor diff(const Tensor& self, int64_t n, int64_t dim, const c10::optional<Tensor>& prepend, const c10::optional<Tensor>& append) {
  diff_check(self, n, dim, prepend, append);
  if (!prepend.has_value() && !append.has_value()) {
    return diff_helper(self, n, dim);
  } else {
    auto a = prepend_append_on_dim(self, prepend, append, dim);
    return diff_helper(a, n, dim);
  }
}

static inline Tensor& diff_out_helper(const Tensor& self, int64_t n, int64_t dim, Tensor& result) {
  auto out_len = self.size(dim) - 1;
  if (self.dtype() == at::kBool) {
    return at::logical_xor_out(result, at::narrow(self, dim, 1, out_len), at::narrow(self, dim, 0, out_len));
  }
  return at::sub_out(result, at::narrow(self, dim, 1, out_len), at::narrow(self, dim, 0, out_len));
}

Tensor& diff_out(const Tensor& self, int64_t n, int64_t dim, const c10::optional<Tensor>& prepend, const c10::optional<Tensor>& append, Tensor& result) {
  diff_check(self, n, dim, prepend, append);
  if (!prepend.has_value() && !append.has_value()) {
    return diff_out_helper(self, n, dim, result);
  } else {
    auto a = prepend_append_on_dim(self, prepend, append, dim);
    return diff_out_helper(a, n, dim, result);
  }
}

void pre_check_gradient(const Tensor& self, c10::optional<int64_t> spacing_size, c10::optional<IntArrayRef> dim,  int64_t edge_order) {
  // Helper for gradient function to make sure input data satisfies prerequisites
  TORCH_CHECK(self.scalar_type() != ScalarType::Byte, "torch.gradient does not support uint8 input.");
  if (spacing_size.has_value() && !dim.has_value()) {
    TORCH_CHECK(spacing_size.value() == 1 || spacing_size.value() == self.dim(), "torch.gradient expected spacing to be unspecified, a scalar or a list of length ", self.dim(), " but got a list of length ", spacing_size.value());
  }
  if (spacing_size.has_value() && dim.has_value()) {
    TORCH_CHECK(spacing_size.value() == dim.value().size(), "torch.gradient expected spacing to be unspecified, a scalar or it's spacing and dim arguments to have the same length, but got a spacing argument of length ", spacing_size.value(), " and a dim argument of length ", dim.value().size(), "." );
  }
  TORCH_CHECK(edge_order == 1 || edge_order == 2, "torch.gradient only supports edge_order=1 and edge_order=2.");
  for (const auto i : c10::irange(self.dim())) {
    TORCH_CHECK(self.size(i) >= edge_order + 1, "torch.gradient expected each dimension size to be at least edge_order+1");
  }
  if (dim.has_value()) {
    // The following function get called to check whether dim argument satisfies prerequisites.
    // The output of the function is not used for the computation of gradient.
    dim_list_to_bitset(dim.value(), self.dim());
  }
}

std::vector<Tensor> gradient_helper(const Tensor& self, TensorList coordinates, IntArrayRef dim, int64_t edge_order) {
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

std::vector<Tensor> gradient_helper_float(const Tensor& self, ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order) {
  std::vector<Tensor> result;
  for (const auto i : c10::irange(dim.size())) {
      int64_t direction = maybe_wrap_dim(dim[i], self.dim());
      auto ax_dx = spacing[i];
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

std::vector<int64_t> gradient_dim_preprocess(const Tensor& self, c10::optional<int64_t> dim) {
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
                       c10::optional<int64_t>(coordinates.size()),
                       c10::optional<IntArrayRef>(dim),
                       edge_order);
    return gradient_helper(self, coordinates, dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, TensorList coordinates, c10::optional<int64_t> dim, int64_t edge_order) {
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  pre_check_gradient(self,
                     c10::optional<int64_t>(coordinates.size()),
                     dim.has_value() ? c10::optional<IntArrayRef>(processed_dim) : c10::nullopt,
                     edge_order);
  return gradient_helper(self, coordinates, processed_dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, c10::ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order) {
  pre_check_gradient(self,
                     c10::optional<int64_t>(spacing.size()),
                     c10::optional<IntArrayRef>(dim),
                     edge_order);
  return gradient_helper_float(self, spacing, dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, ArrayRef<Scalar> spacing, c10::optional<int64_t> dim, int64_t edge_order) {
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  pre_check_gradient(self,
                     c10::optional<int64_t>(spacing.size()),
                     dim.has_value() ? c10::optional<IntArrayRef>(processed_dim) : c10::nullopt,
                     edge_order);
  return gradient_helper_float(self, spacing, processed_dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, const Scalar& unit_size, IntArrayRef dim, int64_t edge_order) {
  // When spacing is given as scalar, while dim is given as IntArrayRef, scalar value need to
  // be taken as unit size at every given dimension element of - dim.
  std::vector<Scalar> spacing(dim.size(), unit_size);
  pre_check_gradient(self,
                     c10::optional<int64_t>(spacing.size()),
                     c10::optional<IntArrayRef>(dim),
                     edge_order);
  return gradient_helper_float(self, spacing, dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, const c10::optional<Scalar>& unit_size, c10::optional<int64_t> dim, int64_t edge_order) {
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  // When unit_size not provided, it is always assumed to be equal to 1.
  // When dim has integer value it implies we are looking for gradient in the specific direction, however when
  // it is not provided, it means we are interested to find gradient in all directions.
  std::vector<Scalar> spacing(dim.has_value() ? 1 : self.dim(),
                              unit_size.has_value() ? unit_size.value() : 1.0) ;
  pre_check_gradient(self,
                     unit_size.has_value() ?  c10::optional<int64_t>(spacing.size()) : c10::nullopt,
                     dim.has_value() ? c10::optional<IntArrayRef>(processed_dim) : c10::nullopt,
                     edge_order);
  return gradient_helper_float(self, spacing, processed_dim, edge_order);
}

std::vector<Tensor> gradient(const Tensor& self, IntArrayRef dim, int64_t edge_order) {
  std::vector<Scalar> spacing(dim.size(), 1.0) ;
  pre_check_gradient(self,
                     c10::optional<int64_t>(spacing.size()),
                     c10::optional<IntArrayRef>(dim),
                     edge_order);
  return gradient_helper_float(self, spacing, dim, edge_order);
}

// ALL REDUCE #################################################################

inline ScalarType get_dtype_from_result(Tensor& result, optional<ScalarType> dtype) {
  TORCH_CHECK(result.defined(), "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
  if (dtype.has_value()) {
    return dtype.value();
  } else {
    return result.scalar_type();
  }
}

inline ScalarType get_dtype_from_self(const Tensor& self, optional<ScalarType> dtype,
                            bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

Tensor& sum_out(const Tensor& self, IntArrayRef dim,
                       bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  auto iter = make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    sum_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor sum(const Tensor &self, c10::optional<ScalarType> dtype) {
  return at::native::sum(self, std::vector<int64_t>{}, false, dtype);
}

Tensor sum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  return at::native::sum_out(self, dim, keepdim, dtype, result);
}

Tensor sum(const Tensor& self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& sum_out(const Tensor& self, DimnameList dim,
                bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  return at::sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

Tensor& nansum_out(const Tensor& self, IntArrayRef dim,
                       bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "nansum does not support complex inputs");
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

Tensor nansum(const Tensor &self, c10::optional<ScalarType> dtype) {
  return at::native::nansum(self, std::vector<int64_t>{}, false, dtype);
}

Tensor nansum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  return at::native::nansum_out(self, dim, keepdim, dtype, result);
}

static Tensor& prod_out_impl(Tensor& result, const Tensor& self, IntArrayRef dim,
                        bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  auto iter = make_reduction("prod", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    prod_stub(iter.device_type(), iter);
  }
  return result;
}

// NOTE: this could be implemented via diag and sum, but this has perf problems,
// see https://github.com/pytorch/pytorch/pull/47305,
Tensor trace_cpu(const Tensor& self) {
  Tensor result;
  // Returns the ScalarType of the self tensor if the tensor is non integral type
  // In the case, self is an integer type tensor, at::kLong is return since promote_integers
  // is set to true
  ScalarType dtype = get_dtype_from_self(self, c10::nullopt, true);
  result = at::empty({}, self.options().dtype(dtype));
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "trace", [&] {
    using accscalar_t = at::acc_type<scalar_t, false>;
    accscalar_t sum = 0;
    const auto* t_data = self.data_ptr<scalar_t>();

    int64_t t_stride_0, t_stride_1, t_diag_size;

    TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());

    t_stride_0 = self.stride(0);
    t_stride_1 = self.stride(1);

    t_diag_size = std::min(self.size(0), self.size(1));
    for (int64_t i = 0; i < t_diag_size; i++) {
      sum += t_data[i * (t_stride_0 + t_stride_1)];
    }

    c10::guts::if_constexpr<std::is_integral<accscalar_t>::value>(
      // all integer types get promoted to kLong
      [&] (auto _) { *result.data_ptr<int64_t>() = _(sum); },  // then-case, invalid for non-integral types
      [&] (auto _) { *result.data_ptr<scalar_t>() = _(sum); }  // else-case, invalid for integral types
    );
  });

  return result;
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  native::prod_out_impl(result, self, dim, keepdim, dtype);
  return result;
}

Tensor prod(const Tensor &self, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, {}, false, dtype);
  return at::native::prod_out_impl(result, self, {}, false, dtype);
}

Tensor& prod_out(const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor& result) {
  return at::native::prod_out_impl(result, self, dim, keepdim, dtype);
}

Tensor prod(const Tensor& self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::prod(self, dimname_to_position(self, dim), keepdim, dtype);
}

Tensor& prod_out(const Tensor& self, Dimname dim,
                 bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  return at::prod_out(result, self, dimname_to_position(self, dim), keepdim, opt_dtype);
}

Tensor &mean_out_cpu_gpu(const Tensor &self, IntArrayRef dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype, Tensor &result) {
  ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");
  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  // TODO: the TensorIterator reduction implementation of mean
  // (mean_kernel_impl()) is unvectorized and leads to very poor performance
  // for production workloads. Once that's fixed, the following code can be used
  // in lieu of the sum + divide implementation below.
  if (self.device().is_cpu()) {
    int64_t dim_prod = 1;
    if (dim.size() == 0 || self.ndimension() == 0) {
      dim_prod = self.numel();
    } else {
      for (auto d : dim) {
        dim_prod *= self.size(d);
      }
    }
    at::sum_out(result, self, dim, keepdim, dtype).div_(dim_prod);
    return result;
  }

  auto iter = make_reduction("mean", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    mean_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor mean_cpu_gpu(const Tensor &self, optional<ScalarType> dtype) {
  return at::native::mean_cpu_gpu(self, IntArrayRef{}, false, dtype);
}

Tensor mean_cpu_gpu(const Tensor& self, IntArrayRef dim, bool keepdim, optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  return at::native::mean_out_cpu_gpu(self, dim, keepdim, dtype, result);
}

Tensor mean(const Tensor& self, DimnameList dim, bool keepdim, optional<ScalarType> dtype) {
  return at::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& mean_out(const Tensor& self, DimnameList dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype, Tensor& result) {
  return at::mean_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

static Tensor squeeze_multiple(const Tensor& self, IntArrayRef dims) {
  int ndims = self.sizes().size();
  auto dims_to_squeeze = at::dim_list_to_bitset(dims, ndims);
  Tensor result = self;
  for (int i = ndims - 1; i >= 0; --i) {
    if (dims_to_squeeze[i]) {
      result = result.squeeze(i);
    }
  }
  return result;
}

static Tensor& logsumexp_out_impl(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  // can't take max of empty tensor
  if (self.numel() != 0) {
    auto maxes = at::amax(self, dims, true);
    auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, dims));
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
  {
    NoNamesGuard guard;
    logsumexp_out_impl(result, self, dims, keepdim);
  }
  namedinference::propagate_names_for_reduction(result, self, dims, keepdim);
  return result;
}

Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::logsumexp_out(self, dims, keepdim, result);
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

static Tensor& norm_out(Tensor &result, const Tensor &self, const optional<Scalar>& opt_p,
                               IntArrayRef dim, bool keepdim, optional<ScalarType> opt_dtype) {
  auto p = opt_p.value_or(2.0).to<double>();
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "norm only supports CPU and CUDA device types, but got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "norm only supports strided layout, but got: ", self.layout());

  ScalarType in_dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(in_dtype) || at::isComplexType(in_dtype),
      "Can only calculate the norm of floating point and complex dtypes. Got ",
      toString(in_dtype),
      " instead.");

  ScalarType out_dtype = result.defined() ? result.scalar_type() : (opt_dtype.has_value() ? opt_dtype.value() : toValueType(self.scalar_type()));

// omit in_dtype in the following call, to avoid make_reduction explicitly casting input to out_dtype
  auto iter = isComplexType(self.scalar_type()) ?
      make_reduction("norm", result, self, dim, keepdim, in_dtype, out_dtype) :
      make_reduction("norm", result, self, dim, keepdim, out_dtype);

  if (iter.numel() == 0) {
    result.zero_();
  } else {
    norm_stub(iter.device_type(), iter, p);
  }
  return result;
}

static inline Tensor _norm(const Tensor &self, const Scalar& p) {
  if (self.is_sparse()) {
    // Sparse tensors need a different implementation because their values
    // are accessed with a different API than strided tensors
    return at::native_norm(self, p);
  } else {
    TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                "norm only supports CPU AND CUDA device type, got: ", self.device().type());
    TORCH_CHECK(self.layout() == Layout::Strided,
                "norm only supports strided layout, got: ", self.layout());
    TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
                "norm only supports floating-point dtypes");

    ScalarType dtype = toValueType(self.scalar_type());
    Tensor result = create_reduction_result(self, IntArrayRef{}, false, dtype);
    return at::native::norm_out(result, self, p, IntArrayRef{}, false, c10::nullopt);
  }
}

Tensor &norm_out(const Tensor& self, const optional<Scalar>& p, IntArrayRef dim, bool keepdim, ScalarType dtype, Tensor& result) {
  return at::native::norm_out(result, self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor &norm_out(const Tensor& self, const optional<Scalar>& p, IntArrayRef dim, bool keepdim, Tensor& result) {
  return at::native::norm_out(result, self, p, dim, keepdim, c10::nullopt);
}

static Tensor norm(const Tensor& self, const optional<Scalar>& p, IntArrayRef dim, bool keepdim,
            optional<ScalarType> opt_dtype) {
  if (self.is_sparse()) {
    // Sparse tensors need a different implementation because their values
    // are accessed with a different API than strided tensors
    return at::native_norm(self, p, dim, keepdim, opt_dtype);
  } else {
    ScalarType out_dtype = value_or_else(opt_dtype, [&] {return toValueType(self.scalar_type());});
    Tensor result = create_reduction_result(self, dim, keepdim, out_dtype);
    return at::native::norm_out(result, self, p, dim, keepdim, opt_dtype);
  }
}

Tensor norm(const Tensor& self, const optional<Scalar>& p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::native::norm(self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, const optional<Scalar>& p, ScalarType dtype) {
  return at::native::norm(self, p, IntArrayRef{}, false, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, const optional<Scalar>& p, IntArrayRef dim, bool keepdim) {
  return at::native::norm(self, p, dim, keepdim, c10::nullopt);
}

// leave it so we support sparse tensors
Tensor norm(const Tensor& self, const Scalar& p) {
  return at::native::_norm(self, p);
}

// Note [all, any : uint8 compatibility]:
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For NumPy comptability, `all` and `any` return
// Tensor of dtype `bool`. However for compatibility reason,
// for `uint8`, they return Tensor of same dtype `uint8`.
// Reference: https://github.com/pytorch/pytorch/pull/47878#issuecomment-747108561
inline const Tensor & _all(const Tensor & result, TensorIterator & iter) {
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    and_stub(iter.device_type(), iter);
  }

  return result;
}

inline TensorIterator get_allany_iter(
    const Tensor& self,
    const Tensor& result,
    IntArrayRef dims,
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

Tensor all(const Tensor& self) {
  Tensor result;

  auto out_dtype =
      meta::check_allany_and_get_output_dtype("all", self, result, false);
  auto shape = meta::get_reduction_shape(self, {}, false);

  result = at::empty(shape, self.options().dtype(out_dtype));
  auto iter = get_allany_iter(self, result, {}, false);

  return _all(result, iter);
}

TORCH_IMPL_FUNC(all_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  dim = maybe_wrap_dim(dim, self.dim());
  auto iter = get_allany_iter(self, result, dim, keepdim);
  auto mut_result = const_cast<Tensor&>(result);
  if (!_dimreduce_return_trivial(mut_result, self, 1, dim, keepdim)) {
    _all(mut_result, iter);
  }
}

inline const Tensor & _any(const Tensor & result, TensorIterator & iter) {
  if (iter.numel() == 0) {
    result.fill_(0);
  } else {
    or_stub(iter.device_type(), iter);
  }

  return result;
}

Tensor any(const Tensor& self) {
  Tensor result;

  auto out_dtype =
      meta::check_allany_and_get_output_dtype("any", self, result, false);
  auto shape = meta::get_reduction_shape(self, {}, false);

  result = at::empty(shape, self.options().dtype(out_dtype));
  auto iter = get_allany_iter(self, result, {}, false);

  return _any(result, iter);
}

TORCH_IMPL_FUNC(any_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  dim = maybe_wrap_dim(dim, self.dim());
  auto iter = get_allany_iter(self, result, dim, keepdim);
  auto mut_result = const_cast<Tensor&>(result);
  if (!_dimreduce_return_trivial(mut_result, self, 0, dim, keepdim)) {
    _any(mut_result, iter);
  }
}

Tensor &amin_out(const Tensor& self, IntArrayRef dim, bool keepdim, Tensor& result) {
  TORCH_CHECK(self.scalar_type() == result.scalar_type(), "Expected the dtype for input and out to match, but got ",
              self.scalar_type(), " for input's dtype and ",  result.scalar_type(), " for out's dtype.");
  if (self.numel() == 0) {
    zero_numel_check_dims(self, dim, "amin()");
  }

  auto iter = make_reduction("amin", result, self, dim, keepdim, self.scalar_type());
  if (iter.numel() != 0) {
    min_values_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor amin(const Tensor& self, IntArrayRef dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::amin_out(result, self, dim, keepdim);
}

Tensor &amax_out(const Tensor& self, IntArrayRef dim, bool keepdim, Tensor& result) {
  TORCH_CHECK(self.scalar_type() == result.scalar_type(), "Expected the dtype for input and out to match, but got ",
              self.scalar_type(), " for input's dtype and ",  result.scalar_type(), " for out's dtype.");
  if (self.numel() == 0) {
    zero_numel_check_dims(self, dim, "amax()");
  }

  auto iter = make_reduction("amax", result, self, dim, keepdim, self.scalar_type());
  if (iter.numel() != 0) {
    max_values_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::amax_out(result, self, dim, keepdim);
}

template <class Stub>
void argmax_argmin_impl(
    const Tensor& self,
    c10::optional<int64_t> dim,
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
 c10::optional<int64_t> dim,
 bool keepdim,
 const Tensor& result) {
  argmax_argmin_impl(self, dim, keepdim, result, argmax_stub);
}

TORCH_IMPL_FUNC(argmin_out)
(const Tensor& self,
 c10::optional<int64_t> dim,
 bool keepdim,
 const Tensor& result) {
  argmax_argmin_impl(self, dim, keepdim, result, argmin_stub);
}

static double std_var_all_cpu(const Tensor& self, int64_t correction, bool take_sqrt) {
  const auto dtype = self.scalar_type();
  TORCH_CHECK(dtype == kDouble || dtype == kFloat,
              "std_var_all: Unsupported dtype ", dtype);

  auto mean = self.mean().item<double>();
  auto iter = TensorIteratorConfig()
      .add_input(self)
      .build();

  auto reduction = [&](int64_t begin, int64_t end, double thread_sum) {
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "std_var_all_cpu", [&] {
      iter.serial_for_each([&] (char** data, const int64_t* strides, int64_t size0, int64_t size1) {
        const double local_mean = mean;
        const int64_t inner_stride = strides[0];
        const int64_t outer_stride = strides[1];

        double local_sum = 0.0;
        for (int64_t i = 0; i < size1; ++i) {
          const char* row_ptr = data[0] + outer_stride * i;
          for (int64_t j = 0; j < size0; ++j) {
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
    return sum_dx2 / std::max(int64_t{0}, self.numel() - correction);
  }();
  const auto result = take_sqrt ? std::sqrt(var) : var;

  if (dtype == kFloat) {
    // Convert to infinity if out of range for a float.
    // Doing it now prevents checked_convert failing later
    return static_cast<float>(result);
  }
  return result;
}

static Tensor& std_var_out(
    const char* fname, Tensor& result, const Tensor& self,
    c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction_opt,
    bool keepdim, bool take_sqrt) {
  TORCH_CHECK(self.device().is_cpu() || self.device().is_cuda(),
              "std and var only supports tensors on a CPU or CUDA device, but got: ",
              self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "std and var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "std and var only support floating point and complex dtypes");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate variance of real and imaginary components
    // seperately then add to get overall variance.
    ScalarType dtype = c10::toValueType(get_dtype_from_result(result, {}));
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
  const auto correction = correction_opt.value_or(1);
  ScalarType dtype = get_dtype_from_result(result, {});
  auto iter = make_reduction(fname, result, self, dim, keepdim, dtype);

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
    c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction_opt,
    bool keepdim, bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              fname, " only supports tensors on a CPU or CUDA device, got: ",
              self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              fname, " only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              fname, " only support floating point and complex dtypes");
  TORCH_CHECK(result1.scalar_type() == c10::toValueType(result2.scalar_type()),
              fname, " expected result1 to be real and match the precision of result2. Got ",
              result1.scalar_type(), " and ", result2.scalar_type(), ".");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate for real and imaginary components seperately then combine as:
    // variance = var_real + var_imag
    // mean = mean_real + j * mean_imag
    ScalarType dtype = c10::toValueType(get_dtype_from_result(result1, {}));
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
  const auto correction = correction_opt.value_or(1);
  ScalarType dtype = get_dtype_from_result(result1, {});
  auto iter =
      make_reduction(fname, result1, result2, self, dim, keepdim, dtype);

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
    const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return at::var_mean(self, /*dim=*/c10::optional<IntArrayRef>(dim),
                      /*correction=*/int64_t{unbiased ? 1 : 0}, keepdim);
}

std::tuple<Tensor, Tensor> std_mean(
    const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return at::std_mean(self, /*dim=*/c10::optional<IntArrayRef>(dim),
                      /*correction=*/int64_t{unbiased ? 1 : 0}, keepdim);
}

std::tuple<Tensor, Tensor> std_mean(const Tensor& self, bool unbiased) {
  return at::std_mean(
      self, /*dim=*/c10::nullopt, /*correction=*/int64_t{unbiased ? 1 : 0});
}

std::tuple<Tensor, Tensor> var_mean(const Tensor& self, bool unbiased) {
  return at::var_mean(
      self, /*dim=*/c10::nullopt, /*correction=*/int64_t{unbiased ? 1 : 0});
}

std::tuple<Tensor&, Tensor&> var_mean_out(
    Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim,
    int64_t correction, bool keepdim) {
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

static TensorOptions options_to_value_type(TensorOptions opts) {
  auto scalar_type = typeMetaToScalarType(opts.dtype());
  return opts.dtype(c10::toValueType(scalar_type));
}

std::tuple<Tensor, Tensor> var_mean(
    const Tensor& self, c10::optional<IntArrayRef> dim,
    c10::optional<int64_t> correction, bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> std_mean(
    const Tensor& self, c10::optional<IntArrayRef> dim,
    c10::optional<int64_t> correction, bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "std_mean", result1, result2, self, dim, correction, keepdim, true);
}

Tensor var(const Tensor& self, bool unbiased) {
  return at::var(
      self, /*dim=*/c10::nullopt, /*correction=*/int64_t{unbiased ? 1 : 0});
}

Tensor var(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return at::var(self, /*dim=*/c10::optional<IntArrayRef>(dim),
                 /*correction=*/int64_t{unbiased ? 1 : 0}, keepdim);
}

Tensor& var_out(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim, Tensor& result) {
  return at::var_out(result, self, /*dim=*/c10::optional<IntArrayRef>(dim),
                     /*correction=*/int64_t{unbiased ? 1 : 0}, keepdim);
}

Tensor std(const Tensor& self, bool unbiased) {
  return at::std(
      self, /*dim=*/c10::nullopt, /*correction=*/int64_t{unbiased ? 1 : 0});
}

Tensor std(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return at::std(self, /*dim=*/c10::optional<IntArrayRef>(dim),
                 /*correction=*/int64_t{unbiased ? 1 : 0}, keepdim);
}

Tensor& std_out(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim, Tensor& result) {
  return at::std_out(result, self, /*dim=*/c10::optional<IntArrayRef>(dim),
                     /*correction=*/int64_t{unbiased ? 1 : 0}, keepdim);
}

Tensor std(const Tensor& self, c10::optional<IntArrayRef> dim,
           c10::optional<int64_t> correction, bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& std_out(
    const Tensor& self, c10::optional<IntArrayRef> dim,
    c10::optional<int64_t> correction, bool keepdim, Tensor& result) {
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& var_out(
    const Tensor& self, c10::optional<IntArrayRef> dim,
    c10::optional<int64_t> correction, bool keepdim, Tensor& result) {
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor var(
    const Tensor& self, c10::optional<IntArrayRef> dim,
    c10::optional<int64_t> correction, bool keepdim) {
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

Tensor std(const Tensor& self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
  return at::std(self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor& std_out(const Tensor& self, DimnameList dim, c10::optional<int64_t> correction,
                bool keepdim, Tensor& result) {
  return at::std_out(result, self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor var(const Tensor& self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor& var_out(const Tensor& self, DimnameList dim, c10::optional<int64_t> correction,
                bool keepdim, Tensor& result) {
  return at::var_out(
      result, self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim,
                                   c10::optional<int64_t> correction, bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim,
                                   c10::optional<int64_t> correction, bool keepdim) {
  return at::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor& norm_out(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& norm_out(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim);
}

Tensor norm(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor norm(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim) {
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
Tensor logcumsumexp(const Tensor& self, Dimname dim) {
  return at::logcumsumexp(self, dimname_to_position(self, dim));
}
Tensor& logcumsumexp_out(const Tensor& self, Dimname dim, Tensor& result) {
  return at::logcumsumexp_out(result, self, dimname_to_position(self, dim));
}
Tensor cumsum(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumsum(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumsum_(Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
    return native::cumsum_(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumsum_out(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype, Tensor& result) {
  return at::cumsum_out(result, self, dimname_to_position(self, dim), dtype);
}
Tensor cumprod(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumprod(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumprod_(Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
    return native::cumprod_(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumprod_out(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype, Tensor& result) {
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
  TORCH_CHECK(self.dtype() == other.dtype(),
              "Expected object of scalar type ", self.dtype(), " but got scalar type ",
              other.dtype(), " for argument 'other'");
  if (!self.is_same_size(other)) {
    return false;
  }
  std::atomic<bool> result{true};
  auto iter = TensorIteratorConfig()
    .add_input(self)
    .add_input(other)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "equal_cpu", [&] {
    iter.for_each([&](char** data, const int64_t *strides, int64_t dim_size) {
      if (!result) {
          return;
      }
      char* self_data = data[0];
      char* other_data = data[1];
      for (int64_t i = 0; i < dim_size; ++i) {
        if (*((scalar_t*)self_data) != *((scalar_t*)other_data)) {
          result = false;
          return;
        }
        self_data += strides[0];
        other_data += strides[1];
      }
    });
  });
  return result.load();
}

// max(dim), min(dim), topk(dim), mode(dim), are examples of reduction
// functions that select values. value_selecting_reduction_backward is the
// backward function for those operators; it propagates the grad to the
// specific value locations referred to at `indices`.
Tensor value_selecting_reduction_backward(const Tensor& grad, int64_t dim, const Tensor& indices, IntArrayRef sizes, bool keepdim) {
  if (!keepdim && sizes.size() > 0) {
    auto grad_ = grad.unsqueeze(dim);
    auto indices_ = indices.unsqueeze(dim);
    return at::zeros(sizes, grad_.options()).scatter_(dim, indices_, grad_);
  }
  return at::zeros(sizes, grad.options()).scatter_(dim, indices, grad);
}

}} // namespace at::native
