#include <gtest/gtest.h>

#include <torch/headeronly/util/order_preserving_flat_hash_map.h>

TEST(TestOrderPreservingFlatHashMap, TestOrderPreservingFlatHashMap) {
  ska_ordered::order_preserving_flat_hash_map<int, int> m;
  m[1] = 10;
  m[2] = 20;
  m[3] = 30;
  EXPECT_EQ(m.size(), 3u);
  EXPECT_EQ(m[2], 20);

  // Insertion order is preserved by iteration.
  int expected_key = 1;
  for (const auto& kv : m) {
    EXPECT_EQ(kv.first, expected_key);
    EXPECT_EQ(kv.second, expected_key * 10);
    ++expected_key;
  }
}
