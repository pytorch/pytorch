#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/core/DispatchKeySet.h>

#include <vector>

using at::DispatchKey;
using at::DispatchKeySet;

TEST(DispatchKeySetTest, TestGetRuntimeDispatchKeySet) {
  // Check if getRuntimeDispatchKeySet and runtimeDispatchKeySetHas agree.
  for (auto dk1: DispatchKeySet(DispatchKeySet::FULL)) {
    auto dks = getRuntimeDispatchKeySet(dk1);
    for (auto dk2: DispatchKeySet(DispatchKeySet::FULL)) {
      ASSERT_EQ(dks.has(dk2), runtimeDispatchKeySetHas(dk1, dk2));
    }
  }
}
