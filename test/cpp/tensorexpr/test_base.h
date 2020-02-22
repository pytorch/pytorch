#pragma once

#include <gtest/gtest.h>
#include <test/cpp/common/support.h>

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
  for (int i = 0; i < v1.size(); i++) {
    EXPECT_NEAR(v1[i], v2[i], threshold)
        << "element index: " << i << ", name: " << name;
  }
}

template <typename T>
static void assertAllEqual(const std::vector<T>& vec, const T& val) {
  for (auto const& elt : vec) {
    ASSERT_EQ(elt, val);
  }
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
