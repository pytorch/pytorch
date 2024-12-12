#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <iostream>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_aminmax_native.h>
#include <ATen/ops/_assert_async_native.h>
#include <ATen/ops/_functional_assert_async_native.h>
#include <ATen/ops/_print_native.h>
#include <ATen/ops/_assert_scalar_native.h>
#include <ATen/ops/_functional_assert_scalar_native.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/allclose_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/argsort_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_max_native.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/clamp_min_native.h>
#include <ATen/ops/clamp_native.h>
#include <ATen/ops/clip_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/fill.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/index.h>
#include <ATen/ops/is_nonzero_native.h>
#include <ATen/ops/isclose.h>
#include <ATen/ops/isclose_native.h>
#include <ATen/ops/isfinite.h>
#include <ATen/ops/isfinite_native.h>
#include <ATen/ops/isin.h>
#include <ATen/ops/isin_native.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isinf_native.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/isneginf_native.h>
#include <ATen/ops/isposinf_native.h>
#include <ATen/ops/isreal_native.h>
#include <ATen/ops/max.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/min.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/mode.h>
#include <ATen/ops/mode_native.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/real.h>
#include <ATen/ops/result_type_native.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/where.h>
#include <ATen/ops/where_native.h>
#include <ATen/ops/zeros_like.h>

#include <iostream>
#include <utility>
#endif

namespace at::meta {

static inline void check_for_unsupported_isin_dtype(const ScalarType type) {
  // Bail out for dtypes unsupported by the sorting algorithm to keep the interface consistent.
  TORCH_CHECK(type != ScalarType::Bool &&
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
  //Manual type promotion, since scalars have to participate in it
  ScalarType result_type = self.scalar_type();
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");
  //Floating is the highest supported
  if (!isFloatingType(result_type)) {
    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(self, state);

    if (min) {
      state = at::native::update_result_type_state(min.get(), state);
    }
    if (max) {
      state = at::native::update_result_type_state(max.get(), state);
    }
    result_type = at::native::result_type(state);
    //disallow type promoting inplace op
    TORCH_CHECK((result_type == self.scalar_type()) ||
       (!(maybe_get_output().defined()) || !(maybe_get_output().is_same(self))),
       "result type ", result_type, " can't be cast to the desired output type ",
       self.dtype());
  }
  //make sure scalars weren't complex
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");
  build_unary_op(maybe_get_output(), self.to(result_type));
}

TORCH_META_FUNC2(clamp, Tensor) (
const Tensor& self,
const OptionalTensorRef min,
const OptionalTensorRef max) {
  TORCH_CHECK(min || max, "torch.clamp: At least one of 'min' or 'max' must not be None");
  TORCH_CHECK(!isComplexType(self.scalar_type()), "clamp is not supported for complex types");
  #define CLAMP_CONFIG()                    \
    TensorIteratorConfig()                  \
      .set_check_mem_overlap(true)          \
      .add_output(maybe_get_output())       \
      .add_const_input(self)                \
      .promote_inputs_to_common_dtype(true) \
      .cast_common_dtype_to_outputs(true)   \
      .enforce_safe_casting_to_output(true)

  if (min && max) {
    build(CLAMP_CONFIG().add_const_input(*min).add_const_input(*max));
  } else if (min) {
    build(CLAMP_CONFIG().add_const_input(*min));
  } else if (max) {
    build(CLAMP_CONFIG().add_const_input(*max));
  }
}


TORCH_META_FUNC(clamp_max) (
  const Tensor& self,
  const Scalar& max
) {
  //we could wrap max into tensor and send to tensor overload,
  //but relu is implemented via clamp_min, so for perf an uniformity reasons
  //do a faster but correct thing
  ScalarType result_type = self.scalar_type();
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");
  TORCH_CHECK(!max.isComplex(), "clamp is not supported for complex types");
  //Floating is the highest supported
  if (!isFloatingType(result_type)) {
    auto result_type = at::native::result_type(self, max);
    TORCH_CHECK((result_type == self.scalar_type()) ||
       (!(maybe_get_output().defined()) || !(maybe_get_output().is_same(self))),
       "result type ", result_type, " can't be cast to the desired output type ",
       self.dtype());
    build_unary_op(maybe_get_output(), self.to(result_type));
  } else {
    build_borrowing_unary_op(maybe_get_output(), self);
  }
}

TORCH_META_FUNC2(clamp_max, Tensor) (
  const Tensor& self,
  const Tensor& max
) {
  build_borrowing_binary_op(maybe_get_output(), self, max);
}


TORCH_META_FUNC(clamp_min) (
  const Tensor& self,
  const Scalar& min
) {
  ScalarType result_type = self.scalar_type();
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");
  TORCH_CHECK(!min.isComplex(), "clamp is not supported for complex types");
  //Floating is the highest supported
  if (!isFloatingType(result_type)) {
    auto result_type = at::native::result_type(self, min);
    TORCH_CHECK((result_type == self.scalar_type() ||
       !(maybe_get_output().defined()) || !(maybe_get_output().is_same(self))),
       "result type ", result_type, " can't be cast to the desired output type ",
       self.dtype());
    build_unary_op(maybe_get_output(), self.to(result_type));
  } else {
    build_borrowing_unary_op(maybe_get_output(), self);
  }
}

TORCH_META_FUNC2(clamp_min, Tensor) (
  const Tensor& self,
  const Tensor& min
) {
  build_borrowing_binary_op(maybe_get_output(), self, min);
}

TORCH_META_FUNC2(isin, Tensor_Tensor) (
  const Tensor& elements, const Tensor& test_elements, bool /*assume_unique*/, bool /*invert*/
) {
  check_for_unsupported_isin_dtype(elements.scalar_type());
  check_for_unsupported_isin_dtype(test_elements.scalar_type());
  set_output_raw_strided(0, elements.sizes(), {}, TensorOptions(elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC2(isin, Tensor_Scalar) (
  const Tensor& elements, const c10::Scalar& test_elements, bool /*assume_unique*/, bool /*invert*/
) {
  check_for_unsupported_isin_dtype(elements.scalar_type());
  check_for_unsupported_isin_dtype(test_elements.type());
  set_output_raw_strided(0, elements.sizes(), {}, TensorOptions(elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC2(isin, Scalar_Tensor) (
  const c10::Scalar& elements, const Tensor& test_elements, bool /*assume_unique*/, bool /*invert*/
) {
  check_for_unsupported_isin_dtype(elements.type());
  check_for_unsupported_isin_dtype(test_elements.scalar_type());
  set_output_raw_strided(0, {0}, {}, TensorOptions(test_elements.device()).dtype(ScalarType::Bool));
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

} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(where_kernel); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(max_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(min_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isposinf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isneginf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(mode_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(clamp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
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
    // For CompositeCompliance, if `other` is a CCT and `self` is a regular Tensor,
    // then we can't perform inplace op into `self` with `other`.
    // NOTE: Inplacing into `close` is fine because it is generated from
    // out-of-place with args `self` and `other`. So if either of them is
    // a CCT then `close` will also be a `CCT`.
    if (isTensorSubclassLike(other)) {
      close.__ior__(self.isnan().bitwise_and(other.isnan()));
    } else {
      close.__ior__(self.isnan().__iand__(other.isnan()));
    }
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


#if !defined(C10_MOBILE)
#define _AT_DISPATCH_INF_TYPES(TYPE, NAME, ...)                          \
        AT_DISPATCH_FLOATING_TYPES_AND3( kHalf, kBFloat16, kFloat8_e5m2, \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_INF_TYPES(TYPE, NAME, ...)           \
        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, \
            TYPE, NAME, __VA_ARGS__)
#endif


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

  return _AT_DISPATCH_INF_TYPES(self.scalar_type(), "isinf", [&]() {
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
    return at::isfinite(at::real(self)).__iand__(at::isfinite(at::imag(self)));
  }

  return _AT_DISPATCH_INF_TYPES(self.scalar_type(), "isfinite", [&]() {
    return (self == self) * (self.abs() != std::numeric_limits<scalar_t>::infinity());
  });
}

void _assert_async_cpu(const Tensor& self) {
  TORCH_CHECK(native::is_nonzero(self), "Expected Tensor with single nonzero value, but got zero");
}

void _assert_async_msg_cpu(const Tensor& self, c10::string_view assert_msg) {
  TORCH_CHECK(native::is_nonzero(self), assert_msg != "" ? assert_msg : "Assertion is failed");
}

void _assert_scalar(const Scalar& scalar, c10::string_view assert_msg) {
  TORCH_SYM_CHECK(scalar.toSymBool(), assert_msg != "" ? assert_msg : "Assertion is failed");
}

Tensor _functional_assert_scalar(const Scalar& scalar, c10::string_view assert_msg, const Tensor& dep_token) {
  _assert_scalar(scalar, assert_msg);
  return dep_token.clone();
}

Tensor _functional_assert_async_msg_cpu(
  const Tensor& self,
  c10::string_view assert_msg,
  const Tensor& dep_token) {
  _assert_async_msg_cpu(self, assert_msg);
  return dep_token.clone();
}

void _print(c10::string_view s) {
  std::cout << s << "\n";
}

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
    std::tie(elements_flat, unique_order) = at::_unique(
        elements, /*sorted=*/ false, /*return_inverse=*/ true);
    std::tie(test_elements_flat, std::ignore) = at::_unique(test_elements, /*sorted=*/ false);
  }

  // 2. Stable sort all elements, maintaining order indices to reverse the
  //    operation. Stable sort is necessary to keep elements before test
  //    elements within the sorted list.
  Tensor all_elements = at::cat({std::move(elements_flat), std::move(test_elements_flat)});
  auto [sorted_elements, sorted_order] = all_elements.sort(
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
    out.copy_(at::index(mask, {std::optional<Tensor>(unique_order)}));
  }
}

template<typename... Args>
Device out_device(Args&... inps){
  for (const auto& i : {inps...}){
    if (i.device() != at::kCPU) {
      return i.device();
    }
  }
  return at::kCPU;
}


Tensor& where_self_out(const Tensor& condition, const Tensor& self, const Tensor& other, Tensor& out) {
  const auto result_type = at::native::result_type(self, other);
  TORCH_CHECK(out.scalar_type() == result_type, "Expected out type to be ", result_type, " but got ", out.scalar_type());

  auto self_ = self.scalar_type() != result_type ? self.to(result_type): self;
  auto other_ = other.scalar_type() != result_type ? other.to(result_type): other;
  auto condition_ = condition;
  auto device = out_device(condition, self_, other_);
  if (device != at::kCPU) { // allow CPU scalars on non-cpu device
    if (condition.device() != device && condition.ndimension() == 0) {
      condition_ = condition.to(device);
    }
    if (self_.device() != device && self_.ndimension() == 0) {
        self_ = self_.to(device);
    }
    if (other_.device() != device && other_.ndimension() == 0) {
        other_ = other_.to(device);
    }
  }
  if (condition_.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE("where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
    condition_ = condition_.to(kBool);
  }
  TORCH_CHECK(condition_.scalar_type() == kBool, "where expected condition to be a boolean tensor, but got a tensor with dtype ", condition_.scalar_type());
  // if there's still a device mismatch, let tensoriterator error out with it
  auto iter = at::TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(out)
    .add_const_input(condition_)
    .add_const_input(self_)
    .add_const_input(other_)
    .build();
  where_kernel(iter.device_type(), iter);
  return out;
}


Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  auto device = out_device(condition, self, other);
  auto result_type = at::native::result_type(self, other);
  Tensor ret = at::empty({0}, self.options().dtype(result_type).device(device));
  at::native::where_self_out(condition, self, other, ret);
  return ret;
}

Tensor where(const Tensor& condition, const Scalar& self, const Tensor& other) {
  auto result_type = at::native::result_type(other, self);
  auto self_converted = at::scalar_tensor(self, other.options().dtype(result_type));
  auto other_converted = other.to(result_type);
  return at::where(condition, self_converted, other_converted);
}

Tensor where(const Tensor& condition, const Tensor& self, const Scalar& other) {
  auto result_type = at::native::result_type(self, other);
  auto other_converted = at::scalar_tensor(other, self.options().dtype(result_type));
  auto self_converted = self.to(result_type);
  return at::where(condition, self_converted, other_converted);
}

Tensor where(const Tensor& condition, const Scalar& self, const Scalar& other) {
  auto result_type = at::native::result_type(self, other);
  const Tensor& other_t = at::scalar_tensor(other, condition.options().dtype(result_type));
  const Tensor& self_t = at::scalar_tensor(self, condition.options().dtype(result_type));
  return at::where(condition, self_t, other_t);
}

std::vector<Tensor> where(const Tensor& condition) {
  return condition.nonzero_numpy();
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::mode_out(self, dim, keepdim, values, indices);
}

std::tuple<Tensor &,Tensor &> mode_out(const Tensor& self, int64_t dim, bool keepdim,
                                       Tensor& values, Tensor& indices) {
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda() || self.is_xpu(),
              "mode only supports CPU, CUDA and XPU device type, got: ", self.device().type());
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
  TORCH_CHECK(self.qscheme() == at::kPerTensorAffine, "Max operator for quantized tensors only works for per tensor quantized tensors. "
  "Please open an issue on https://github.com/pytorch/pytorch/issues if you need per channel quantized tensor support.");
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor max = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
  at::max_outf(self.int_repr(), dim, keepdim, max, max_indices);
  // TODO: qscheme
  return std::tuple<Tensor, Tensor>(
      at::_make_per_tensor_quantized_tensor(max, self.q_scale(), self.q_zero_point()), max_indices);
}

std::tuple<Tensor, Tensor> qmin(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.qscheme() == at::kPerTensorAffine, "Min operator for quantized tensors only works for per tensor quantized tensors. "
  "Please open an issue on https://github.com/pytorch/pytorch/issues if you need per channel quantized tensor support.");
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
 const Tensor& /*self*/,
 const OptionalScalarRef min,
 const OptionalScalarRef max,
 const Tensor& result) {
  using at::native::detail::ClampLimits;
  if (min && max) {
    if (min.get().toDouble() != min.get().toDouble() ||
        max.get().toDouble() != max.get().toDouble()) {
      at::fill_(const_cast<Tensor&>(result), std::numeric_limits<double>::quiet_NaN());
    } else {
      clamp_scalar_stub(device_type(), *this, min.get(), max.get());
    }
  } else if (max) {
    clamp_max_scalar_stub(device_type(), *this, max.get());
  } else if (min) {
    clamp_min_scalar_stub(device_type(), *this, min.get());
  }
}

TORCH_IMPL_FUNC(clamp_Tensor_out)
(const Tensor& self, const OptionalTensorRef min,
                  const OptionalTensorRef max, const Tensor&) {
  if (min && max) {
    clamp_stub(device_type(), *this);
  } else if (min) {
    maximum_stub(device_type(), *this);
  } else if (max) {
    minimum_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(clamp_max_out)
(const Tensor& self, const Scalar& max, const Tensor& result) {
  if (max.toDouble() != max.toDouble()) {
//TODO this is not great, building TI again is expensive, but I can't use
//fill_stub because fill is not structured
//this is a corner case anyway
    at::fill_(const_cast<Tensor&>(result), wrapped_scalar_tensor(max));
  } else {
    clamp_max_scalar_stub(device_type(), *this, max);
  }
}

TORCH_IMPL_FUNC(clamp_max_Tensor_out)
(const Tensor& self, const Tensor& max, const Tensor& result) {
  minimum_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(clamp_min_out)
(const Tensor& self, const Scalar& min, const Tensor& result) {
  if (min.toDouble() != min.toDouble()) {
    at::fill_(const_cast<Tensor&>(result), min);
  } else {
    clamp_min_scalar_stub(device_type(), *this, min);
  }
}

TORCH_IMPL_FUNC(clamp_min_Tensor_out)
(const Tensor& self, const Tensor& min, const Tensor& result) {
  maximum_stub(device_type(), *this);
}

// Implements the "clip" alias for clamp
Tensor& clip_out(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max, Tensor& result) {
  return at::clamp_outf(self, min, max, result);
}

Tensor& clip_out(const Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max, Tensor& result) {
  return at::clamp_outf(self, min, max, result);
}

Tensor clip(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
  return at::clamp(self, min, max);
}

Tensor clip(const Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max) {
  return at::clamp(self, min, max);
}

Tensor& clip_(Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
  return at::clamp_(self, min, max);
}

Tensor& clip_(Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max) {
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
Tensor argsort(const Tensor& /*self*/, Dimname /*dim*/, bool /*keepdim*/) {
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

} // namespace at::native
