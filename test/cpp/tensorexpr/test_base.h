#pragma once

#if defined(USE_GTEST)
#include <gtest/gtest.h>
#include <test/cpp/common/support.h>
#else
#include <cmath>
#include "c10/util/Exception.h"
#include "test/cpp/tensorexpr/gtest_assert_float_eq.h"
#define ASSERT_EQ(x, y, ...) TORCH_INTERNAL_ASSERT((x) == (y), __VA_ARGS__)
#define ASSERT_FLOAT_EQ(x, y, ...) \
  TORCH_INTERNAL_ASSERT(AlmostEquals((x), (y)), __VA_ARGS__)
#define ASSERT_NE(x, y, ...) TORCH_INTERNAL_ASSERT((x) != (y), __VA_ARGS__)
#define ASSERT_GT(x, y, ...) TORCH_INTERNAL_ASSERT((x) > (y), __VA_ARGS__)
#define ASSERT_GE(x, y, ...) TORCH_INTERNAL_ASSERT((x) >= (y), __VA_ARGS__)
#define ASSERT_LT(x, y, ...) TORCH_INTERNAL_ASSERT((x) < (y), __VA_ARGS__)
#define ASSERT_LE(x, y, ...) TORCH_INTERNAL_ASSERT((x) <= (y), __VA_ARGS__)

#define ASSERT_NEAR(x, y, a, ...) \
  TORCH_INTERNAL_ASSERT(std::fabs((x) - (y)) < (a), __VA_ARGS__)

#define ASSERT_TRUE TORCH_INTERNAL_ASSERT
#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))
#define ASSERT_THROWS_WITH(statement, substring)                         \
  try {                                                                  \
    (void)statement;                                                     \
    ASSERT_TRUE(false);                                                  \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }
#define ASSERT_ANY_THROW(statement)     \
  {                                     \
    bool threw = false;                 \
    try {                               \
      (void)statement;                  \
    } catch (const std::exception& e) { \
      threw = true;                     \
    }                                   \
    ASSERT_TRUE(threw);                 \
  }

#endif // defined(USE_GTEST)

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename U, typename V>
void ExpectAllNear(
    const std::vector<U>& v1,
    const std::vector<U>& v2,
    V threshold,
    const std::string& name = "") {
  ASSERT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); i++) {
    ASSERT_NEAR(v1[i], v2[i], threshold);
  }
}

template <typename U, typename V>
void ExpectAllNear(
    const std::vector<U>& vec,
    const U& val,
    V threshold,
    const std::string& name = "") {
  for (size_t i = 0; i < vec.size(); i++) {
    ASSERT_NEAR(vec[i], val, threshold);
  }
}

template <typename T>
static void assertAllEqual(const std::vector<T>& vec, const T& val) {
  for (auto const& elt : vec) {
    ASSERT_EQ(elt, val);
  }
}

template <typename T>
static void assertAllEqual(const std::vector<T>& v1, const std::vector<T>& v2) {
  ASSERT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_EQ(v1[i], v2[i]);
  }
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
