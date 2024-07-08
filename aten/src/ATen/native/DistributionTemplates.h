#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/Generator.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Tensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Optional.h>
#include <limits>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native::templates {

// ==================================================== Random ========================================================

// The purpose of `update_from` and `update_to` is to find the closest valid int64_t number that can be used as actual `from`.
// The current implementation of `random_` uses uint64_t arithmetics and casts the result to the target dtype(scalar_t).
// This casting can result in generating numbers that happen to be greater or equal to `to` value. For instance:
//
//    auto actual = torch::empty({3, 3}, torch::half);
//    actual.random_(0, 65504);
//
// If random's uint64_t arithmetics produces 65503 as a random value after casting to torch::half it becomes 65504
// and violates the requirement that random value must be less than `to`. To resolve this issue `update_from` and `update_to`
// moves `from` to the right and `to` to the left to the next closest value that won't go outside [from, to) after casting to
// the target dtype. For `to` = 65504 it moves left for (1 << (log2(to) - 11 + 1)) = 32 and becomes 65472, which is previous
// available number for torch::half dtype.
template<typename scalar_t>
int64_t update_from(int64_t from) {
  static_assert(
    std::is_floating_point<scalar_t>::value ||
    std::is_same<scalar_t, at::Half>::value ||
    std::is_same<scalar_t, at::BFloat16>::value, "scalar_t must be floating-point type");
  const auto from_plus_1 = static_cast<int64_t>(static_cast<scalar_t>(from + 1));
  if (from_plus_1 < from) {
    int64_t from_ = std::abs(from + 1);
    int n = 0;
    while (from_ >>= 1) ++n;
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    from = from_plus_1 + (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return from;
}

template<typename scalar_t>
int64_t update_to(int64_t to) {
  static_assert(
    std::is_floating_point<scalar_t>::value ||
    std::is_same<scalar_t, at::Half>::value ||
    std::is_same<scalar_t, at::BFloat16>::value, "scalar_t must be floating-point type");
  const auto to_minus_1 = static_cast<int64_t>(static_cast<scalar_t>(to - 1));
  if (to_minus_1 >= to) {
    int64_t to_ = std::abs(to - 1);
    int n = 0;
    while (to_ >>= 1) ++n;
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return to;
}

// Return earlier for not invoking kernel.
// See https://github.com/pytorch/pytorch/issues/103418 for more details
#define CHECK_EMPTY_AND_RETURN(tensor) \
  if (tensor.numel() == 0) {  \
    return tensor;  \
  }

template<template<typename> class random_kernel, typename RNG>
at::Tensor& random_impl(at::Tensor& self, std::optional<Generator> generator) {
  CHECK_EMPTY_AND_RETURN(self);
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  random_kernel<RNG>()(iter, generator);
  return self;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name , " is out of bounds for ", dtype); \

#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype) \
  if (var < -(1LL << digits) || var > (1LL << digits)) { \
    TORCH_WARN(name , " is out of bounds [-(2^", digits, "), 2^", digits, "]. ", \
      "Due to precision limitations ", dtype, " can support discrete uniform distribution only within this range. ", \
      "This warning will become an error in version 1.7 release, please fix the code in advance"); \
  }

static void check_from_to_in_range(int64_t from, int64_t to_inc, caffe2::TypeMeta dtype) {
  const auto scalar_type = typeMetaToScalarType(dtype);
  if (isFloatingType(scalar_type)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "check_random_fp_bounds", [&] {
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);

      constexpr auto digits = std::numeric_limits<scalar_t>::digits;
      WARN_OUT_OF_BOUNDS(from, "from", digits, dtype);
      WARN_OUT_OF_BOUNDS(to_inc, "to - 1", digits, dtype);
    });
  } else if (scalar_type == kUInt64) {
    // When you do a comparison between int64_t and uint64_t, the usual
    // arithmetic conversions say that the int64_t value is promoted to
    // unsigned. But this conversion wraps around: if I had -1 as my int64_t,
    // then it will promote to 0xFFFFFFFFFFFFFFFF in uint64_t. This is never
    // the right thing to do.
    CHECK_OUT_OF_BOUNDS(from, "from", 0, INT64_MAX, dtype);
    CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", 0, INT64_MAX, dtype);
  } else if (isIntegralType(scalar_type, /*includeBool=*/true)) {
    AT_DISPATCH_V2(scalar_type, "check_random_integral_bounds", AT_WRAP([&]() {
      const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);
    }), AT_EXPAND(AT_INTEGRAL_TYPES), kUInt16, kUInt32, kBool);
  } else {
    TORCH_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");
  }
}

template<template<typename> class random_from_to_kernel, typename RNG>
at::Tensor& random_from_to_impl(at::Tensor& self, int64_t from, std::optional<int64_t> to_opt, std::optional<Generator> generator) {
  uint64_t range = 0;
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  if (to_opt.has_value()) {
    // [from, to)
    int64_t to = *to_opt;
    TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_update_from_to", [&] {
        from = update_from<scalar_t>(from);
        to = update_to<scalar_t>(to);
        TORCH_CHECK(from < to, "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
      });
    }
    check_from_to_in_range(from, to - 1, self.dtype());
    CHECK_EMPTY_AND_RETURN(self);
    range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    random_from_to_kernel<RNG>()(iter, range, from, generator);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    int64_t to_inc = 0;
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_from_to_range_calc", [&] {
        constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
        to_inc = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(scalar_t_max);
        from = update_from<scalar_t>(from);
        TORCH_CHECK(from < to_inc, "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=", from, " > to_inc=", to_inc);
      });
    } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
      AT_DISPATCH_V2(self.scalar_type(), "random_from_to_range_calc", AT_WRAP([&] {
        if constexpr (std::is_same_v<scalar_t, bool>) {
          to_inc = static_cast<int64_t>(true);
        } else {
          to_inc = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        }
      }), AT_EXPAND(AT_INTEGRAL_TYPES_V2), kBool);
    } else {
      TORCH_CHECK(false, "random_from_to_impl handles only integral, floating-point and boolean types");
    }
    check_from_to_in_range(from, to_inc, self.dtype());
    CHECK_EMPTY_AND_RETURN(self);
    range = static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
    random_from_to_kernel<RNG>()(iter, range, from, generator);
  } else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64
    CHECK_EMPTY_AND_RETURN(self);
    random_from_to_kernel<RNG>()(iter, generator);
  }
  return self;
}

// ==================================================== Normal ========================================================

#define CHECK_NORMAL_TENSOR_STD(std) \
  do { \
    TORCH_CHECK( \
      !std.is_complex(), \
      "normal expects standard deviation to be non-complex"); \
    TORCH_CHECK( \
      std.numel() == 0 || std.is_meta() || std.min().ge(0).item<bool>(), \
      "normal expects all elements of std >= 0.0"); \
  } while (0)

#define CHECK_NORMAL_STD(std) \
  TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_impl_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  CHECK_NORMAL_STD(std);
  CHECK_EMPTY_AND_RETURN(self);

  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    // variance for normal distribution of the real and imaginary values
    // is half of the input variance
    normal_kernel<RNG>()(float_tensor, mean, std/(std::sqrt(2)), gen);
  } else {
    normal_kernel<RNG>()(self, mean, std, gen);
  }
  return self;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, const Tensor& mean, double std, std::optional<Generator> gen) {
  CHECK_NORMAL_STD(std);
  auto std_tensor = at::empty_like(output, MemoryFormat::Contiguous);
  auto shape = at::infer_size(mean.sizes(), std_tensor.sizes());
  at::native::resize_output(output, shape);
  normal_impl_<normal_kernel, RNG>(output, 0, std, gen);
  output.add_(mean);
  return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, double mean, const Tensor& std, std::optional<Generator> gen) {
  CHECK_NORMAL_TENSOR_STD(std);
  auto mean_tensor = at::full({}, mean, output.options());
  auto shape = at::infer_size(mean_tensor.sizes(), std.sizes());
  at::native::resize_output(output, shape);
  normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
  // CUDA NB: addcmul_out copies the tensor to be added into the output.
  // The previous function here was addcmul_out(output, mean_tensor, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean_tensor + mean_tensor * std instead of mean_tensor + output * std
  output.mul_(std).add_(mean_tensor);
  return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  CHECK_NORMAL_TENSOR_STD(std);
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  at::native::resize_output(output, shape);
  normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
  // CUDA NB: addcmul_out copies the tensor to be added into the output.
  // The previous function here was addcmul_out(output, mean, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean + mean * std instead of mean + output * std
  output.mul_(std).add_(mean);
  return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, double std, std::optional<Generator> gen) {
  CHECK_NORMAL_STD(std);
  Tensor ret = at::empty_like(mean, MemoryFormat::Contiguous);
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  return ret;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(double mean, const Tensor& std, std::optional<Generator> gen) {
  CHECK_NORMAL_TENSOR_STD(std);
  Tensor ret = at::empty_like(std, MemoryFormat::Contiguous);
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  return ret;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  CHECK_NORMAL_TENSOR_STD(std);
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  Tensor ret = at::empty(shape, mean.options(), MemoryFormat::Contiguous);
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  return ret;
}

// ==================================================== Uniform =======================================================

template<template<typename> class uniform_kernel, typename RNG>
at::Tensor& uniform_impl_(at::Tensor& self, double from, double to, std::optional<Generator> generator) {
  if (self.is_complex()) {
    CHECK_EMPTY_AND_RETURN(self);
    auto float_tensor = at::view_as_real(self);
    uniform_impl_<uniform_kernel, RNG>(float_tensor, from, to, generator);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "check_uniform_bounds", [&] {
      const auto dtype = self.dtype();
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to, "to", min, max, dtype);
      TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
      TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
            "uniform_ expects to-from <= std::numeric_limits<", toString(self.scalar_type()),
            ">::max(), but found to=", to, " and from=", from,
            " which result in to-from to exceed the limit");
      from = std::min(std::max(from, min), max);
      to = std::max(std::min(to, max), min);
    });
    CHECK_EMPTY_AND_RETURN(self);
    auto iter = at::TensorIterator::borrowing_nullary_op(self);
    uniform_kernel<RNG>()(iter, from, to, generator);
  }
  return self;
}

// ================================================== LogNormal =======================================================

template<template<typename> class log_normal_kernel, typename RNG>
at::Tensor& log_normal_impl_(at::Tensor& self, double mean, double std, std::optional<Generator> gen) {
  TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
  CHECK_EMPTY_AND_RETURN(self);
  auto iter = TensorIterator::borrowing_nullary_op(self);
  log_normal_kernel<RNG>()(iter, mean, std, gen);
  return self;
}

// =================================================== Geometric ======================================================

template<template<typename> class geometric_kernel, typename RNG>
Tensor& geometric_impl_(Tensor& self, double p, std::optional<Generator> gen) {
  TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
  CHECK_EMPTY_AND_RETURN(self);
  auto iter = TensorIterator::borrowing_nullary_op(self);
  geometric_kernel<RNG>()(iter, p, gen);
  return self;
}

// ================================================== Exponential =====================================================

template<template<typename> class exponential_kernel, typename RNG>
Tensor& exponential_impl_(Tensor& self, double lambda, std::optional<Generator> gen) {
  TORCH_CHECK(lambda > 0.0, "exponential_ expects lambda > 0.0, but found lambda=", lambda);
  CHECK_EMPTY_AND_RETURN(self);
  auto iter = TensorIterator::borrowing_nullary_op(self);
  exponential_kernel<RNG>()(iter, lambda, gen);
  return self;
}

// ==================================================== Cauchy ========================================================

template<template<typename> class cauchy_kernel, typename RNG>
Tensor& cauchy_impl_(Tensor& self, double median, double sigma, std::optional<Generator> gen) {
  // TODO: instead of variable name 'sigma', use 'gamma' or 'scale'
  // the variance, squared sigma, is undefined for cauchy distribution
  TORCH_CHECK(sigma > 0.0, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "Cauchy distribution is a continuous probability distribution. dtype must be a floating point but you specified ", self.dtype());
  CHECK_EMPTY_AND_RETURN(self);
  auto iter = TensorIterator::borrowing_nullary_op(self);
  cauchy_kernel<RNG>()(iter, median, sigma, gen);
  return self;
}

// ==================================================== Bernoulli =====================================================

template<template<typename> class bernoulli_tensor_kernel, typename RNG>
Tensor& bernoulli_impl_(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  CHECK_EMPTY_AND_RETURN(self);
  NoNamesGuard guard;
  at::assert_no_internal_overlap(self);
  bernoulli_tensor_kernel<RNG>()(self, p_, gen);
  return self;
}

template<template<typename> class bernoulli_scalar_kernel, typename RNG>
Tensor& bernoulli_impl_(Tensor& self, double p, std::optional<Generator> gen) {
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  CHECK_EMPTY_AND_RETURN(self);
  at::assert_no_internal_overlap(self);
  bernoulli_scalar_kernel<RNG>()(self, p, gen);
  return self;
}

template<template<typename> class bernoulli_tensor_kernel, typename RNG>
Tensor& bernoulli_out_impl(Tensor& result, const Tensor& self, std::optional<Generator> gen) {
  // result.resize_as_(self) requires self to have same dtype as result, so we
  // use resize_ instead.
  // TODO: Fix resize_as_. See pytorch/pytorch#11665.
  result.resize_(self.sizes());
  bernoulli_impl_<bernoulli_tensor_kernel, RNG>(result, self, gen);
  namedinference::propagate_names(result, self);
  return result;
}

#undef CHECK_OUT_OF_BOUNDS
#undef WARN_OUT_OF_BOUNDS

} // namespace at::native::templates
