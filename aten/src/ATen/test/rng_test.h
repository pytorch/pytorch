#include <gtest/gtest.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <stdexcept>

namespace test {

template<template<typename> class random_kernel, typename RNG>
at::Tensor& random_(at::Tensor& self, at::Generator* generator) {
  auto gen = (RNG*)generator;
  auto iter = at::TensorIterator::nullary_op(self);
  random_kernel<RNG>()(iter, gen);
  return self;
}

template<template<typename> class random_from_to_kernel, typename RNG>
at::Tensor& random_from_to(at::Tensor& self, int64_t from, c10::optional<int64_t> to, at::Generator* generator) {
  auto gen = (RNG*)generator;
  uint64_t range;
  auto iter = at::TensorIterator::nullary_op(self);
  if (to) {
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

template<typename RNG, c10::ScalarType S, typename T>
void test_random_from_to(const at::Device& device) {
  const auto t_min_val = std::numeric_limits<T>::lowest();
  const auto int64_min_val = std::numeric_limits<int64_t>::lowest();
  const int64_t min_val = std::is_floating_point<T>::value ? int64_min_val : static_cast<int64_t>(t_min_val);

  const auto t_max_val = std::numeric_limits<T>::max();
  const auto int64_max_val = std::numeric_limits<int64_t>::max();
  const int64_t max_val = std::is_floating_point<T>::value ? int64_max_val : static_cast<int64_t>(t_max_val);

  const auto uint64_max_val = std::numeric_limits<uint64_t>::max();

  std::vector<int64_t> froms;
  std::vector<c10::optional<int64_t>> tos;
  if (std::is_same<T, bool>::value) {
    froms = {
      0L
    };
    tos = {
      1L,
      static_cast<c10::optional<int64_t>>(c10::nullopt)
    };
  } else if (std::is_signed<T>::value) {
    froms = {
      min_val,
      -42L,
      0L,
      42L
    };
    tos = {
      c10::optional<int64_t>(-42L),
      c10::optional<int64_t>(0L),
      c10::optional<int64_t>(42L),
      c10::optional<int64_t>(max_val),
      static_cast<c10::optional<int64_t>>(c10::nullopt)
    };
  } else {
    froms = {
      0L,
      42L
    };
    tos = {
      c10::optional<int64_t>(42L),
      c10::optional<int64_t>(max_val),
      static_cast<c10::optional<int64_t>>(c10::nullopt)
    };
  }

  const std::vector<uint64_t> vals = {
    0L,
    42L,
    static_cast<uint64_t>(max_val),
    static_cast<uint64_t>(max_val) + 1,
    uint64_max_val
  };

  bool full_64_bit_range_case_covered = false;
  bool from_to_case_covered = false;
  bool from_case_covered = false;
  for (const int64_t from : froms) {
    for (const c10::optional<int64_t> to : tos) {
      if (!to.has_value() || from < *to) {
        for (const uint64_t val : vals) {
          auto gen = new RNG(val);

          auto actual = torch::empty({3, 3}, torch::TensorOptions().dtype(S).device(device));
          actual.random_(from, to, gen);

          T exp;
          uint64_t range;
          if (!to.has_value() && from == int64_min_val) {
            exp = val;
            full_64_bit_range_case_covered = true;
          } else {
            if (to.has_value()) {
              range = *to - from;
              from_to_case_covered = true;
            } else {
              range = max_val - from + 1;
              from_case_covered = true;
            }
            if (range < (1ULL << 32)) {
              exp = static_cast<T>(static_cast<int64_t>((static_cast<uint32_t>(val) % range + from)));
            } else {
              exp = static_cast<T>(static_cast<int64_t>((val % range + from)));
            }
          }
          ASSERT_TRUE(from <= exp);
          if (to) {
            ASSERT_TRUE(static_cast<int64_t>(exp) < *to);
          }
          const auto expected = torch::full_like(actual, exp);
          std::cout << "actual = " << actual << std::endl;
          std::cout << "expected = " << expected << std::endl;
          if (std::is_same<T, bool>::value) {
            ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
          } else {
            ASSERT_TRUE(torch::allclose(actual, expected));
          }
        }
      }
    }
  }
  if (std::is_same<T, int64_t>::value) {
    ASSERT_TRUE(full_64_bit_range_case_covered);
  }
  ASSERT_TRUE(from_to_case_covered);
  ASSERT_TRUE(from_case_covered);
}

template<typename RNG, c10::ScalarType S, typename T>
void test_random(const at::Device& device) {
  const auto min_val = std::numeric_limits<T>::lowest();
  const auto max_val = std::numeric_limits<T>::max();
  const auto uint64_max_val = std::numeric_limits<uint64_t>::max();

  const std::vector<uint64_t> vals = {
    0L,
    42L,
    static_cast<uint64_t>(max_val),
    static_cast<uint64_t>(max_val) + 1,
    uint64_max_val
  };

  for (const uint64_t val : vals) {
    auto gen = new RNG(val);

    auto actual = torch::empty({3, 3}, torch::TensorOptions().dtype(S).device(device));
    actual.random_(gen);

    uint64_t range;
    if (std::is_floating_point<T>::value) {
      range = static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1);
    } else if (std::is_same<T, bool>::value) {
      range = 2;
    } else {
      range = static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1;
    }
    T exp;
    if (std::is_same<T, double>::value || std::is_same<T, int64_t>::value) {
      exp = val % range;
    } else {
      exp = static_cast<uint32_t>(val) % range;
    }

    ASSERT_TRUE(0 <= static_cast<int64_t>(exp));
    ASSERT_TRUE(static_cast<int64_t>(exp) < range);

    const auto expected = torch::full_like(actual, exp);
    if (std::is_same<T, bool>::value) {
      ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
    } else {
      ASSERT_TRUE(torch::allclose(actual, expected));
    }
  }
}

}
