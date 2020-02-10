#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Optional.h>
#include <limits>

namespace at {
namespace native {
namespace templates {

template<template<typename> class random_kernel, typename RNG>
at::Tensor& random_impl(at::Tensor& self, at::Generator* generator) {
  auto gen = (RNG*)generator;
  auto iter = at::TensorIterator::nullary_op(self);
  random_kernel<RNG>()(iter, gen);
  return self;
}

template<template<typename> class random_from_to_kernel, typename RNG>
at::Tensor& random_from_to_impl(at::Tensor& self, int64_t from, c10::optional<int64_t> to, at::Generator* generator) {
  auto gen = (RNG*)generator;
  uint64_t range;
  auto iter = at::TensorIterator::nullary_op(self);
  if (to.has_value()) {
    // [from, to)
    TORCH_CHECK(from < *to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", *to);
    range = *to - from;
    random_from_to_kernel<RNG>()(iter, range, from, gen);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "random_from_to_range_calc", [&] {
      if (std::is_same<scalar_t, bool>::value) {
        range = 2;
      } else {
        const auto t_max_val = std::numeric_limits<scalar_t>::max();
        const auto int64_max_val = std::numeric_limits<int64_t>::max();
        const int64_t max_val = std::is_floating_point<scalar_t>::value ? int64_max_val : static_cast<int64_t>(t_max_val);
        range = max_val - from + 1;
      }
    });
    random_from_to_kernel<RNG>()(iter, range, from, gen);
  } else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64
    random_from_to_kernel<RNG>()(iter, gen);
  }
  return self;
}

}}}
