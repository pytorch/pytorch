#include <gtest/gtest.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>
#include <optional>
#include <torch/all.h>
#include <stdexcept>

namespace {

constexpr auto int64_min_val = std::numeric_limits<int64_t>::lowest();
constexpr auto int64_max_val = std::numeric_limits<int64_t>::max();
template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _min_val() {
  return int64_min_val;
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _min_val() {
  return static_cast<int64_t>(std::numeric_limits<T>::lowest());
}

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _min_from() {
  return -(static_cast<int64_t>(1) << std::numeric_limits<T>::digits);
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _min_from() {
  return _min_val<T>();
}

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _max_val() {
  return int64_max_val;
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _max_val() {
  return static_cast<int64_t>(std::numeric_limits<T>::max());
}

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr int64_t _max_to() {
  return static_cast<int64_t>(1) << std::numeric_limits<T>::digits;
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr int64_t _max_to() {
  return _max_val<T>();
}

template<typename RNG, c10::ScalarType S, typename T>
void test_random_from_to(const at::Device& device) {

  constexpr int64_t max_val = _max_val<T>();
  constexpr int64_t max_to = _max_to<T>();

  constexpr auto uint64_max_val = std::numeric_limits<uint64_t>::max();

  std::vector<int64_t> froms;
  std::vector<::std::optional<int64_t>> tos;
  if constexpr (::std::is_same_v<T, bool>) {
    froms = {
      0L
    };
    tos = {
      1L,
      static_cast<::std::optional<int64_t>>(::std::nullopt)
    };
  } else if constexpr (::std::is_signed_v<T>) {
    constexpr int64_t min_from = _min_from<T>();
    froms = {
      min_from,
      -42L,
      0L,
      42L
    };
    tos = {
      ::std::optional<int64_t>(-42L),
      ::std::optional<int64_t>(0L),
      ::std::optional<int64_t>(42L),
      ::std::optional<int64_t>(max_to),
      static_cast<::std::optional<int64_t>>(::std::nullopt)
    };
  } else {
    froms = {
      0L,
      42L
    };
    tos = {
      ::std::optional<int64_t>(42L),
      ::std::optional<int64_t>(max_to),
      static_cast<::std::optional<int64_t>>(::std::nullopt)
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
    for (const ::std::optional<int64_t> & to : tos) {
      if (!to.has_value() || from < *to) {
        for (const uint64_t val : vals) {
          auto gen = at::make_generator<RNG>(val);

          auto actual = torch::empty({3, 3}, torch::TensorOptions().dtype(S).device(device));
          actual.random_(from, to, gen);

          T exp;
          uint64_t range;
          if (!to.has_value() && from == int64_min_val) {
            exp = static_cast<int64_t>(val);
            full_64_bit_range_case_covered = true;
          } else {
            if (to.has_value()) {
              range = static_cast<uint64_t>(*to) - static_cast<uint64_t>(from);
              from_to_case_covered = true;
            } else {
              range = static_cast<uint64_t>(max_to) - static_cast<uint64_t>(from) + 1;
              from_case_covered = true;
            }
#ifdef FBCODE_CAFFE2
            if (range < (1ULL << 32)) {
#else
            // this is leaking details of implementation into test
            // we are starting to use random64() at 2^28 to minimize skew due to %
            if (range < (1ULL << 28)) {
#endif
              exp = static_cast<T>(static_cast<int64_t>((static_cast<uint32_t>(val) % range + from)));
            } else {
              exp = static_cast<T>(static_cast<int64_t>((val % range + from)));
            }
          }
          ASSERT_TRUE(from <= exp);
          if (to.has_value()) {
            ASSERT_TRUE(static_cast<int64_t>(exp) < *to);
          }
          const auto expected = torch::full_like(actual, exp);
          if constexpr (::std::is_same_v<T, bool>) {
            ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
          } else {
            ASSERT_TRUE(torch::allclose(actual, expected));
          }
        }
      }
    }
  }
  if constexpr (::std::is_same_v<T, int64_t>) {
    ASSERT_TRUE(full_64_bit_range_case_covered);
  } else {
    (void)full_64_bit_range_case_covered;
  }
  ASSERT_TRUE(from_to_case_covered);
  ASSERT_TRUE(from_case_covered);
}

template<typename RNG, c10::ScalarType S, typename T>
void test_random(const at::Device& device) {
  const auto max_val = _max_val<T>();
  const auto uint64_max_val = std::numeric_limits<uint64_t>::max();

  const std::vector<uint64_t> vals = {
    0L,
    42L,
    static_cast<uint64_t>(max_val),
    static_cast<uint64_t>(max_val) + 1,
    uint64_max_val
  };

  for (const uint64_t val : vals) {
    auto gen = at::make_generator<RNG>(val);

    auto actual = torch::empty({3, 3}, torch::TensorOptions().dtype(S).device(device));
    actual.random_(gen);

    uint64_t range;
    if constexpr (::std::is_floating_point_v<T>) {
      range = static_cast<uint64_t>((1ULL << ::std::numeric_limits<T>::digits) + 1);
    } else if constexpr (::std::is_same_v<T, bool>) {
      range = 2;
    } else {
      range = static_cast<uint64_t>(::std::numeric_limits<T>::max()) + 1;
    }
    T exp;
    if constexpr (::std::is_same_v<T, double> || ::std::is_same_v<T, int64_t>) {
      exp = val % range;
    } else {
      exp = static_cast<uint32_t>(val) % range;
    }

    ASSERT_TRUE(0 <= static_cast<int64_t>(exp));
    ASSERT_TRUE(static_cast<uint64_t>(exp) < range);

    const auto expected = torch::full_like(actual, exp);
    if constexpr (::std::is_same_v<T, bool>) {
      ASSERT_TRUE(torch::allclose(actual.toType(torch::kInt), expected.toType(torch::kInt)));
    } else {
      ASSERT_TRUE(torch::allclose(actual, expected));
    }
  }
}

}
