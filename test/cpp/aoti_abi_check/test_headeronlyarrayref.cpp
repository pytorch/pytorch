#include <gtest/gtest.h>

#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

#include <vector>

using torch::headeronly::HeaderOnlyArrayRef;

TEST(TestHeaderOnlyArrayRef, TestEmpty) {
  HeaderOnlyArrayRef<float> arr;
  ASSERT_TRUE(arr.empty());
}

TEST(TestHeaderOnlyArrayRef, TestSingleton) {
  float val = 5.0f;
  HeaderOnlyArrayRef<float> arr(val);
  ASSERT_FALSE(arr.empty());
  EXPECT_EQ(arr.size(), 1);
  EXPECT_EQ(arr[0], val);
}

TEST(TestHeaderOnlyArrayRef, TestAPIs) {
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7};
  HeaderOnlyArrayRef<int> arr(vec);
  ASSERT_FALSE(arr.empty());
  EXPECT_EQ(arr.size(), 7);
  for (size_t i = 0; i < arr.size(); i++) {
    EXPECT_EQ(arr[i], i + 1);
    EXPECT_EQ(arr.at(i), i + 1);
  }
  EXPECT_EQ(arr.front(), 1);
  EXPECT_EQ(arr.back(), 7);
  ASSERT_TRUE(arr.slice(3, 4).equals(arr.slice(3)));
}

TEST(TestHeaderOnlyArrayRef, TestFromInitializerList) {
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7};
  HeaderOnlyArrayRef<int> arr({1, 2, 3, 4, 5, 6, 7});
  auto res_vec = arr.vec();
  for (size_t i = 0; i < vec.size(); i++) {
    EXPECT_EQ(vec[i], res_vec[i]);
  }
}

TEST(TestHeaderOnlyArrayRef, TestFromRange) {
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7};
  HeaderOnlyArrayRef<int> arr(vec.data() + 3, vec.data() + 7);
  auto res_vec = arr.vec();
  for (size_t i = 0; i < res_vec.size(); i++) {
    EXPECT_EQ(vec[i + 3], res_vec[i]);
  }
}
