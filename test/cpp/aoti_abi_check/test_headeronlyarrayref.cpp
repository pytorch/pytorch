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

TEST(TestHeaderOnlyArrayRef, TestEqualityOperators) {
  std::vector<int> vec = {1, 2, 3, 4};
  std::vector<int> same_vec = {1, 2, 3, 4};
  std::vector<int> diff_vec = {1, 2, 3, 5};
  std::vector<int> short_vec = {1, 2, 3};

  HeaderOnlyArrayRef<int> arr(vec);
  HeaderOnlyArrayRef<int> same_arr(same_vec);
  HeaderOnlyArrayRef<int> diff_arr(diff_vec);
  HeaderOnlyArrayRef<int> short_arr(short_vec);

  // HeaderOnlyArrayRef vs HeaderOnlyArrayRef
  EXPECT_TRUE(arr == same_arr);
  EXPECT_FALSE(arr != same_arr);
  EXPECT_FALSE(arr == diff_arr);
  EXPECT_TRUE(arr != diff_arr);
  EXPECT_FALSE(arr == short_arr);
  EXPECT_TRUE(arr != short_arr);

  // HeaderOnlyArrayRef should agree with .equals()
  EXPECT_EQ(arr == same_arr, arr.equals(same_arr));
  EXPECT_EQ(arr == diff_arr, arr.equals(diff_arr));

  // HeaderOnlyArrayRef vs std::vector (both orderings)
  EXPECT_TRUE(arr == same_vec);
  EXPECT_TRUE(same_vec == arr);
  EXPECT_FALSE(arr != same_vec);
  EXPECT_FALSE(same_vec != arr);

  EXPECT_FALSE(arr == diff_vec);
  EXPECT_FALSE(diff_vec == arr);
  EXPECT_TRUE(arr != diff_vec);
  EXPECT_TRUE(diff_vec != arr);
}
