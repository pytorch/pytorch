#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Optional.h>
#include <limits>
#include <cmath>

namespace at {
namespace native {
namespace templates {

// The purpose of `update_from` and `update_to` is to find the closest valid int64_t number that can be used as actual `from`.
// The current implementation of `random_` uses uint64_t arithmetics and casts the result to the target dtype(scalar_t).
// This casting can result in generating numbers that happen to be greater or equal to `to` value. For instance:
//
//    auto actual = torch::empty({3, 3}, torch::half);
//    actual.random_(0, 65504);
//
// If random's uint64_t arithmetics produces 65503 as a random value after casting to torch::half it becomes 65504
// and violates the requirement that random value must be less than `to`. To resolve this issue `update_from` and `update_to`
// moves `from` to the left and `to` to the right to the next closest value that won't go outside [from, to) after casting to
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
    to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return to;
}

template<template<typename> class random_kernel, typename RNG>
at::Tensor& random_impl(at::Tensor& self, at::Generator* generator) {
  auto gen = (RNG*)generator;
  auto iter = at::TensorIterator::nullary_op(self);
  random_kernel<RNG>()(iter, gen);
  return self;
}

template<template<typename> class random_from_to_kernel, typename RNG>
at::Tensor& random_from_to_impl(at::Tensor& self, int64_t from, c10::optional<int64_t> to_opt, at::Generator* generator) {
  auto gen = (RNG*)generator;
  uint64_t range = 0;
  auto iter = at::TensorIterator::nullary_op(self);
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
    range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    random_from_to_kernel<RNG>()(iter, range, from, gen);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    int64_t to_inc = 0;
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_from_to_range_calc", [&] {
        to_inc = std::numeric_limits<scalar_t>::max() > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        from = update_from<scalar_t>(from);
        TORCH_CHECK(from < to_inc, "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=", from, " > to_inc=", to_inc);
      });
    } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
      AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "random_from_to_range_calc", [&] {
        if (std::is_same<scalar_t, bool>::value) {
          to_inc = static_cast<int64_t>(true);
        } else {
          to_inc = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        }
      });
    } else {
      TORCH_CHECK(false, "random_from_to_impl handles only integral, floating-point and boolean types");
    }
    range = static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
    random_from_to_kernel<RNG>()(iter, range, from, gen);
  } else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64
    random_from_to_kernel<RNG>()(iter, gen);
  }
  return self;
}

}}}
