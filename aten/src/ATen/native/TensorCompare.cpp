#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorIndexing.h>

namespace at {
namespace meta {

static inline void check_for_unsupported_isin_dtype(const ScalarType type) {
  // Bail out for dtypes unsupported by the sorting algorithm to keep the interface consistent.
  TORCH_CHECK(type != ScalarType::Bool &&
      type != ScalarType::BFloat16 &&
      type != ScalarType::ComplexFloat &&
      type != ScalarType::ComplexDouble,
      "Unsupported input type encountered for isin(): ", type);
}

TORCH_META_FUNC(clamp) (
const Tensor& self,
const OptionalScalarRef min,
const OptionalScalarRef max) {
  if (!min && !max) {
    TORCH_CHECK(false, "torch.clamp: At least one of 'min' or 'max' must not be None");
  }

  build_borrowing_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC2(isin, Tensor_Tensor) (
  const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert
) {
  check_for_unsupported_isin_dtype(elements.scalar_type());
  check_for_unsupported_isin_dtype(test_elements.scalar_type());
  set_output(elements.sizes(), TensorOptions(elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC2(isin, Tensor_Scalar) (
  const Tensor& elements, const c10::Scalar& test_elements, bool assume_unique, bool invert
) {
  check_for_unsupported_isin_dtype(elements.scalar_type());
  check_for_unsupported_isin_dtype(test_elements.type());
  set_output(elements.sizes(), TensorOptions(elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC2(isin, Scalar_Tensor) (
  const c10::Scalar& elements, const Tensor& test_elements, bool assume_unique, bool invert
) {
  check_for_unsupported_isin_dtype(elements.type());
  check_for_unsupported_isin_dtype(test_elements.scalar_type());
  set_output({0}, TensorOptions(test_elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC(isposinf) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "isposinf does not support complex inputs.");
  TORCH_CHECK(maybe_get_output().defined() ? maybe_get_output().dtype() == at::kBool : true,
              "isposinf does not support non-boolean outputs.");
  build_borrowing_unary_force_boolean_op(maybe_get_output(), self);
}

TORCH_META_FUNC(isneginf) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "isneginf does not support complex inputs.");
  TORCH_CHECK(maybe_get_output().defined() ? maybe_get_output().dtype() == at::kBool : true,
              "isneginf does not support non-boolean outputs.");
  build_borrowing_unary_force_boolean_op(maybe_get_output(), self);
}

static void check_unsupported_complex(const char* name, const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), name, ": does not support complex input");
}

TORCH_PRECOMPUTE_META_FUNC2(max, dim)
(const Tensor& self, int64_t dim, bool keepdim) {
  dim = maybe_wrap_dim(dim, self.dim());
  at::native::zero_numel_check_dims(self, dim, "max()");
  check_unsupported_complex("max()", self);
  resize_reduction_with_indices(*this, self, dim, keepdim, self.scalar_type());
  return TORCH_PRECOMPUTE_STRUCT2(max, dim)()
      .set_dim(maybe_wrap_dim(dim, self.dim()));
}

TORCH_PRECOMPUTE_META_FUNC2(min, dim)(const Tensor& self, int64_t dim, bool keepdim) {
  dim = maybe_wrap_dim(dim, self.dim());
  at::native::zero_numel_check_dims(self, dim, "min()");
  check_unsupported_complex("min()", self);
  resize_reduction_with_indices(*this, self, dim, keepdim, self.scalar_type());
  return TORCH_PRECOMPUTE_STRUCT2(min, dim)()
      .set_dim(maybe_wrap_dim(dim, self.dim()));
}

} // namespace meta

namespace native {

DEFINE_DISPATCH(where_kernel); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(max_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(min_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isposinf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isneginf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(mode_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(clamp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(clamp_min_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(clamp_max_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(clamp_scalar_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(clamp_min_scalar_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(clamp_max_scalar_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isin_default_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  return at::isclose(self, other, rtol, atol, equal_nan).all().item<uint8_t>();
}

// Note [closeness]
// A number A is close to B when either:
//
// (1) A is equal to B, with NaNs comparing equal when equal_nan is true.
// (2) The error abs(A - B) is finite and less than the max error
//      (atol + abs(rtol * B)).
//
// Note that this is consistent with NumPy's isclose but divergent from
// Python's isclose, which computes the max error symmetrically as
// max(rtol * max(abs(A), abs(B)), atol).
// TODO: use bitwise operator overloads once we add them
// TODO: revisit complex inputs and equal_nan=true after
//  https://github.com/numpy/numpy/issues/15959 is resolved
Tensor isclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type());
  TORCH_CHECK(!(self.is_quantized() || other.is_quantized()),
    "isclose is not supported for quantized inputs.");

  // Checks that rtol and atol are non-negative
  // Note: consistent with Python's isclose but divergent from NumPy's, which
  //  allows negative atol and rtol.
  TORCH_CHECK(rtol >= 0, "rtol must be greater than or equal to zero, but got ", rtol);
  TORCH_CHECK(atol >= 0, "atol must be greater than or equal to zero, but got ", atol);

  // Computes equality closeness
  Tensor close = self == other;
  if (equal_nan && (self.is_floating_point() || self.is_complex())) {
      close.__ior__(self.isnan().__iand__(other.isnan()));
  }

  // In case of zero tolerances the closeness inequality degenerates to an equality check.
  // In this case, the short-circuit prevents false positives as detailed in the paragraph below.
  if (rtol == 0 && atol == 0){
      return close;
  }

  // Note [closeness error computation]
  // atol and rtol are provided as doubles, so the computation
  // rtol * other will produce a float or complex tensor.
  // When the difference (self - other) is compared to it then the
  // tensor representing the difference will also be cast to float or complex.
  // However, since (self - other) in uint8 is very likely to produce a
  // negative value, this moves the cast forward so the difference is
  // always computed in a float or complex type.
  // If the values of the integer tensors cannot be exactly represented
  // by the default scalar type then this may cause an incorrect result.

  // Computes allowed and actual error
  Tensor cast_self, cast_other;
  cast_self = self.scalar_type() == at::kBool ? self.to(at::get_default_dtype()) : self;
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    cast_other = other.to(at::get_default_dtype());
  } else {
    cast_other = other;
  }

  Tensor allowed_error = atol + (rtol * cast_other).abs();
  Tensor actual_error = (cast_self - cast_other).abs();

  // Computes finite closeness
  close.__ior__(at::isfinite(actual_error).__iand__(actual_error <= allowed_error));

  return close;
}

Tensor isnan(const Tensor& self) {
  return self != self;
}

Tensor isreal(const Tensor& self) {
  // Note: Integral and Floating tensor values are always real
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) ||
      c10::isFloatingType(self.scalar_type())) {
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  return at::imag(self) == 0;
}

Tensor isinf(const Tensor &self) {
  // Note: Integral tensor values are never infinite
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    return at::zeros_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // Note: a complex value is infinite when either part is infinite
  if (self.is_complex()) {
    return at::isinf(at::real(self)).__ior__
          (at::isinf(at::imag(self)));
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "isinf", [&]() {
    return self.abs() == std::numeric_limits<scalar_t>::infinity();
  });
}

Tensor isfinite(const Tensor& self) {
  // Note: Integral tensor values are always finite
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // Note: a complex value is finite iff both parts are finite
  if (self.is_complex()) {
    return at::isfinite(self.abs());
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "isfinite", [&]() {
    return (self == self) * (self.abs() != std::numeric_limits<scalar_t>::infinity());
  });
}

void _assert_async_cpu(const Tensor& self) {
  TORCH_CHECK(native::is_nonzero(self), "Expected Tensor with single nonzero value, but got zero");
}

namespace {

// DO NOT USE THIS -- it's just an implementation detail of wrapped_scalar tensor below.
at::Tensor scalar_to_tensor_default_dtype(
    const Scalar& s,
    const Device device = at::kCPU) {
  if (s.isFloatingPoint()) {
    return at::scalar_tensor(
        s, at::device(device).dtype(at::get_default_dtype()));
  } else if (s.isBoolean()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kBool));
  } else if (s.isComplex()) {
    return at::scalar_tensor(
        s, at::device(device).dtype(at::get_default_complex_dtype()));
  } else {
    TORCH_INTERNAL_ASSERT(s.isIntegral(false));
    return at::scalar_tensor(s, at::device(device).dtype(at::kLong));
  }
}

// TLDR: Don't call `wrapped_scalar_tensor_default_dtype` -- this function is only necessary to support the partial
// type-promotion that torch.where supports.  Once torch.where fully supports type promotion, we
// won't need this function.
//
// Longer explanation:
// `wrapped_scalar_tensor_default_dtype` is a bit of a hack because torch.where doesn't support type promotion, but
// does support `torch.where(tensor, scalar1, scalar2)` with default scalar types.  The trickiness is we
// usually convert double scalars to doubles, and `set_wrapped_number` defines type promotion priority
// as being below tensor types rather than as the default dtype (perhaps we should?).  This wouldn't matter
// if we just supported type normal type promotion on torch.where, however.
Tensor wrapped_scalar_tensor_default_dtype(
    const Scalar& scalar,
    Device device) {
  at::Tensor tensor;
  tensor = scalar_to_tensor_default_dtype(scalar, device);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

} // anonymous namespace

// Sorting-based algorithm for isin(); used when the number of test elements is large.
static void isin_sorting(
    const Tensor& elements,
    const Tensor& test_elements,
    bool assume_unique,
    bool invert,
    const Tensor& out) {
  // 1. Concatenate unique elements with unique test elements in 1D form. If
  //    assume_unique is true, skip calls to unique().
  Tensor elements_flat, test_elements_flat, unique_order;
  if (assume_unique) {
    elements_flat = elements.ravel();
    test_elements_flat = test_elements.ravel();
  } else {
    std::tie (elements_flat, unique_order) = at::_unique(
        elements, /*sorted=*/ false, /*return_inverse=*/ true);
    std::tie (test_elements_flat, std::ignore) = at::_unique(test_elements, /*sorted=*/ false);
  }

  // 2. Stable sort all elements, maintaining order indices to reverse the
  //    operation. Stable sort is necessary to keep elements before test
  //    elements within the sorted list.
  Tensor all_elements = at::cat({elements_flat, test_elements_flat});
  Tensor sorted_elements, sorted_order;
  std::tie (sorted_elements, sorted_order) = all_elements.sort(
      /*stable=*/ true, /*dim=*/ 0, /*descending=*/ false);

  // 3. Create a mask for locations of adjacent duplicate values within the
  //    sorted list. Duplicate values are in both elements and test elements.
  Tensor duplicate_mask = at::empty_like(sorted_elements, TensorOptions(ScalarType::Bool));
  Tensor sorted_except_first = sorted_elements.slice(0, 1, at::indexing::None);
  Tensor sorted_except_last = sorted_elements.slice(0, 0, -1);
  duplicate_mask.slice(0, 0, -1).copy_(
    invert ? sorted_except_first.ne(sorted_except_last) : sorted_except_first.eq(sorted_except_last));
  duplicate_mask.index_put_({-1}, invert);

  // 4. Reorder the mask to match the pre-sorted element order.
  Tensor mask = at::empty_like(duplicate_mask);
  mask.index_copy_(0, sorted_order, duplicate_mask);

  // 5. Index the mask to match the pre-unique element order. If
  //    assume_unique is true, just take the first N items of the mask,
  //    where N is the original number of elements.
  if (assume_unique) {
    out.copy_(mask.slice(0, 0, elements.numel()).view_as(out));
  } else {
    out.copy_(at::index(mask, {unique_order}));
  }
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
              "Expected condition, x and y to be on the same device, but condition is on ",
              condition.device(), " and x and y are on ", self.device(), " and ", other.device(),
              " respectively");

  if (condition.scalar_type() == ScalarType::Byte) {
  TORCH_WARN_ONCE("where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
} else {
  TORCH_CHECK(condition.scalar_type() == ScalarType::Bool, "where expected condition to be a boolean tensor, but got a tensor with dtype ", condition.scalar_type());
}

  c10::MaybeOwned<Tensor> b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
  return at::_s_where(*b_condition, *b_self, *b_other);
}

Tensor where(const Tensor& condition, const Scalar& self, const Tensor& other) {
  return at::where(condition, wrapped_scalar_tensor(self, other.device()), other);
}

Tensor where(const Tensor& condition, const Tensor& self, const Scalar& other) {
  return at::where(condition, self, wrapped_scalar_tensor(other, self.device()));
}

Tensor where(const Tensor& condition, const Scalar& self, const Scalar& other) {
  const auto device = condition.device();
  const Tensor& other_t = wrapped_scalar_tensor_default_dtype(other, device);
  const Tensor& self_t = wrapped_scalar_tensor_default_dtype(self, device);
  return at::where(condition, self_t, other_t);
}

std::vector<Tensor> where(const Tensor& condition) {
  return condition.nonzero_numpy();
}

Tensor _s_where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());
  Tensor ret = at::empty(self.sizes(), self.options());
  //
  Tensor cond_bool = condition.scalar_type() == ScalarType::Byte ? condition.to(ScalarType::Bool) : condition;
  auto iter = at::TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(ret)
    .add_input(cond_bool)
    .add_input(self)
    .add_input(other)
    .build();
  where_kernel(iter.device_type(), iter);
  return ret;
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::mode_out(self, dim, keepdim, values, indices);
}

std::tuple<Tensor &,Tensor &> mode_out(const Tensor& self, int64_t dim, bool keepdim,
                                       Tensor& values, Tensor& indices) {
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "mode only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "mode only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == values.device(),
              "expected device '", self.device(), "' but got '",
              values.device(), "' for values output");
  TORCH_CHECK(self.device() == indices.device(),
              "expected device '", self.device(), "' but got '",
              indices.device(), "' for indices output");
  TORCH_CHECK(self.scalar_type() == values.scalar_type(),
              "expected scalar type '", self.scalar_type(), "' but got '",
              values.scalar_type(), "' for values output");
  TORCH_CHECK(indices.scalar_type() == ScalarType::Long,
              "expected scalar type '", ScalarType::Long, "' but got '",
              indices.scalar_type(), "' for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    auto sizes = get_zero_numel_tensor_size(self, dim, keepdim, "mode()");
    resize_output(values, sizes);
    resize_output(indices, sizes);
    return std::tie(values, indices);
  }
  else if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    auto result = [&]() {
      NoNamesGuard guard;
      mode_stub(self.device().type(), values, indices, self, dim, keepdim);
      return std::tuple<Tensor &,Tensor &>{values, indices};
    }();
    namedinference::propagate_names_for_reduction(std::get<0>(result), self, dim, keepdim);
    namedinference::propagate_names_for_reduction(std::get<1>(result), self, dim, keepdim);
    return result;
  }
}

template <class Stub>
void minmax_out_impl(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const Tensor& values,
    const Tensor& indices,
    Stub& stub) {
  NoNamesGuard guard;
  if (self.numel() > 0) {
    if (self.numel() == 1 && self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else {
      stub(self.device().type(), values, indices, self, dim, keepdim);
    }
  }
}

TORCH_IMPL_FUNC(max_out)
(const Tensor& self,
 int64_t dim,
 bool keepdim,
 const Tensor& values,
 const Tensor& indices) {
  minmax_out_impl(self, dim, keepdim, values, indices, max_stub);
}

TORCH_IMPL_FUNC(min_out)
(const Tensor& self,
 int64_t dim,
 bool keepdim,
 const Tensor& values,
 const Tensor& indices) {
  minmax_out_impl(self, dim, keepdim, values, indices, min_stub);
}

std::tuple<Tensor, Tensor> qmax(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor max = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
  at::max_outf(self.int_repr(), dim, keepdim, max, max_indices);
  // TODO: qscheme
  return std::tuple<Tensor, Tensor>(
      at::_make_per_tensor_quantized_tensor(max, self.q_scale(), self.q_zero_point()), max_indices);
}

std::tuple<Tensor, Tensor> qmin(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor min_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor min = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
  at::min_outf(self.int_repr(), dim, keepdim, min, min_indices);
  return std::tuple<Tensor, Tensor>(
      at::_make_per_tensor_quantized_tensor(min, self.q_scale(), self.q_zero_point()), min_indices);
}

// DEPRECATED: Use at::aminmax instead
std::tuple<Tensor, Tensor> _aminmax(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_WARN_ONCE("_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
                  " This warning will only appear once per process.");
  return at::aminmax(self, dim, keepdim);
}

TORCH_IMPL_FUNC(clamp_out)
(
 const Tensor& self,
 const OptionalScalarRef min,
 const OptionalScalarRef max,
 const Tensor& result) {
  using at::native::detail::ClampLimits;
  if (min && max) {
    clamp_scalar_stub(device_type(), *this, min.get(), max.get());
  } else if (max) {
    clamp_max_scalar_stub(device_type(), *this, max.get());
  } else if (min) {
    clamp_min_scalar_stub(device_type(), *this, min.get());
  }
}

Tensor& clamp_out(const Tensor& self, const c10::optional<Tensor>& min,
                  const c10::optional<Tensor>& max, Tensor& result) {
  if (min && max) {
    TORCH_CHECK(self.layout() == Layout::Strided,
                "torch.clamp only supports strided layout, got: ", self.layout());
    auto iter = TensorIteratorConfig()
                .set_check_mem_overlap(true)
                .add_output(result)
                .add_input(self)
                .add_input(*min)
                .add_input(*max)
                .promote_inputs_to_common_dtype(true)
                .cast_common_dtype_to_outputs(true)
                .enforce_safe_casting_to_output(true)
                .build();
    clamp_stub(iter.device_type(), iter);
  } else if (max) {
    at::clamp_max_outf(self, *max, result);
  } else if (min) {
    at::clamp_min_outf(self, *min, result);
  } else {
    TORCH_CHECK(false, "torch.clamp: At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor clamp(const Tensor& self, const c10::optional<Scalar>& min, const c10::optional<Scalar>& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_outf(self, min, max, result);
}

Tensor clamp(const Tensor& self, const c10::optional<Tensor>& min, const c10::optional<Tensor>& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_outf(self, min, max, result);
}

Tensor& clamp_(Tensor& self, const c10::optional<Scalar>& min, const c10::optional<Scalar>& max) {
  return at::clamp_outf(self, min, max, self);
}

Tensor& clamp_(Tensor& self, const c10::optional<Tensor>& min, const c10::optional<Tensor>& max) {
  return at::clamp_outf(self, min, max, self);
}

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  clamp_max_scalar_stub(iter.device_type(), iter, max);
  return result;
}

Tensor& clamp_max_out(const Tensor& self, const Tensor& max, Tensor& result) {
  TORCH_CHECK(self.layout() == Layout::Strided,
              "torch.clamp only supports strided layout, got: ", self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, max);
  clamp_max_stub(iter.device_type(), iter);
  return result;
}

Tensor clamp_max(const Tensor& self, const Scalar& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_max_outf(self, max, result);
}

Tensor clamp_max(const Tensor& self, const Tensor& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_max_outf(self, max, result);
}

Tensor& clamp_max_(Tensor& self, const Scalar& max) {
  return at::clamp_max_outf(self, max, self);
}

Tensor& clamp_max_(Tensor& self, const Tensor& max) {
  return at::clamp_max_outf(self, max, self);
}

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  clamp_min_scalar_stub(iter.device_type(), iter, min);
  return result;
}

Tensor& clamp_min_out(const Tensor& self, const Tensor& min, Tensor& result) {
  TORCH_CHECK(self.layout() == Layout::Strided,
              "torch.clamp only supports strided layout, got: ", self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, min);
  clamp_min_stub(iter.device_type(), iter);
  return result;
}

Tensor clamp_min(const Tensor& self, const Scalar& min) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_min_outf(self, min, result);
}

Tensor clamp_min(const Tensor& self, const Tensor& min) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_min_outf(self, min, result);
}

Tensor& clamp_min_(Tensor& self, const Scalar& min) {
  return at::clamp_min_outf(self, min, self);
}

Tensor& clamp_min_(Tensor& self, const Tensor& min) {
  return at::clamp_min_outf(self, min, self);
}

// Implements the "clip" alias for clamp
Tensor& clip_out(const Tensor& self, const c10::optional<Scalar>& min, const c10::optional<Scalar>& max, Tensor& result) {
  return at::clamp_outf(self, min, max, result);
}

Tensor& clip_out(const Tensor& self, const c10::optional<Tensor>& min, const c10::optional<Tensor>& max, Tensor& result) {
  return at::clamp_outf(self, min, max, result);
}

Tensor clip(const Tensor& self, const c10::optional<Scalar>& min, const c10::optional<Scalar>& max) {
  return at::clamp(self, min, max);
}

Tensor clip(const Tensor& self, const c10::optional<Tensor>& min, const c10::optional<Tensor>& max) {
  return at::clamp(self, min, max);
}

Tensor& clip_(Tensor& self, const c10::optional<Scalar>& min, const c10::optional<Scalar>& max) {
  return at::clamp_(self, min, max);
}

Tensor& clip_(Tensor& self, const c10::optional<Tensor>& min, const c10::optional<Tensor>& max) {
  return at::clamp_(self, min, max);
}

// Named tensor overloads

std::tuple<Tensor, Tensor> min(const Tensor& self, Dimname dim, bool keepdim) {
  return at::min(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> min_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& min, Tensor& min_indices) {
  return at::min_out(min, min_indices, self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor, Tensor> max(const Tensor& self, Dimname dim, bool keepdim) {
  return at::max(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor&, Tensor&> max_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& max, Tensor& max_indices) {
  return at::max_out(max, max_indices, self, dimname_to_position(self, dim), keepdim);
}
Tensor argmax(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmax");
}
Tensor argmin(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmin");
}
Tensor argsort(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argsort");
}
std::tuple<Tensor, Tensor> mode(const Tensor& self, Dimname dim, bool keepdim) {
  return at::mode(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> mode_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& values, Tensor& indices) {
  return at::mode_out(values, indices, self, dimname_to_position(self, dim), keepdim);
}

TORCH_IMPL_FUNC(isin_Tensor_Tensor_out) (
  const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out
) {
  if (elements.numel() == 0) {
    return;
  }

  // Heuristic taken from numpy's implementation.
  // See https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575
  if (test_elements.numel() < static_cast<int64_t>(
        10.0f * std::pow(static_cast<double>(elements.numel()), 0.145))) {
    out.fill_(invert);
    isin_default_stub(elements.device().type(), elements, test_elements, invert, out);
  } else {
    isin_sorting(elements, test_elements, assume_unique, invert, out);
  }
}

TORCH_IMPL_FUNC(isin_Tensor_Scalar_out) (
  const Tensor& elements, const c10::Scalar& test_elements, bool assume_unique, bool invert, const Tensor& out
) {
  // redispatch to eq / ne
  if (invert) {
    at::ne_out(const_cast<Tensor&>(out), elements, test_elements);
  } else {
    at::eq_out(const_cast<Tensor&>(out), elements, test_elements);
  }
}

TORCH_IMPL_FUNC(isin_Scalar_Tensor_out) (
  const c10::Scalar& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out
) {
  // redispatch
  at::isin_out(const_cast<Tensor&>(out), wrapped_scalar_tensor(elements, test_elements.device()),
    test_elements, assume_unique, invert);
}

TORCH_IMPL_FUNC(isposinf_out) (const Tensor& self, const Tensor& result) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    result.fill_(false);
  } else {
    isposinf_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(isneginf_out) (const Tensor& self, const Tensor& result) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    result.fill_(false);
  } else {
    isneginf_stub(device_type(), *this);
  }
}

} // namespace native
} // namespace at
