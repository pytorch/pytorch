#include <ATen/native/cudnn/LRUCache.h>

#include <gtest/gtest.h>

#include <array>
#include <memory>

namespace {

using at::native::LRUCache;

TEST(CudnnLRUCacheTest, LeastRecentlyUsed) {
  LRUCache<int, int> cache(2);
  int hits = 0;
  int misses = 0;
  int evictions = 0;
  for (int key : std::array{1, 2, 1, 3, 1, 2}) {
    if (auto* value = cache.find(key)) {
      ++hits;
      EXPECT_EQ(*value, key * 10);
    } else {
      ++misses;
      evictions += cache.update(key, key * 10);
    }
    EXPECT_LE(cache.size(), 2);
  }
  EXPECT_EQ(hits, 2);
  EXPECT_EQ(misses, 4);
  EXPECT_EQ(evictions, 2);
  EXPECT_NE(cache.find(1), nullptr);
  EXPECT_NE(cache.find(2), nullptr);
  EXPECT_EQ(cache.find(3), nullptr);
}

TEST(CudnnLRUCacheTest, ReplacementPromotes) {
  LRUCache<int, std::unique_ptr<int>> cache(2);
  EXPECT_FALSE(cache.update(1, std::make_unique<int>(10)));
  EXPECT_FALSE(cache.update(2, std::make_unique<int>(20)));
  EXPECT_FALSE(cache.update(1, std::make_unique<int>(30)));
  EXPECT_TRUE(cache.update(3, std::make_unique<int>(40)));
  EXPECT_EQ(cache.size(), 2);
  ASSERT_NE(cache.find(1), nullptr);
  EXPECT_EQ(**cache.find(1), 30);
  EXPECT_EQ(cache.find(2), nullptr);
}

TEST(CudnnLRUCacheTest, Unlimited) {
  LRUCache<int, int> cache(0);
  for (int key = 0; key < 128; ++key) {
    EXPECT_FALSE(cache.update(key, key));
  }
  EXPECT_FALSE(cache.update(0, 200));
  EXPECT_EQ(cache.size(), 128);
  EXPECT_EQ(*cache.find(0), 200);
  EXPECT_EQ(*cache.find(127), 127);
}

TEST(CudnnLRUCacheTest, Disabled) {
  LRUCache<int, std::shared_ptr<int>> cache(-1);
  auto value = std::make_shared<int>(10);
  EXPECT_FALSE(cache.update(1, value));
  EXPECT_EQ(cache.find(1), nullptr);
  EXPECT_EQ(cache.size(), 0);
  EXPECT_EQ(value.use_count(), 1);
}

} // namespace
