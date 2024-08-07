#include <gtest/gtest.h>

#include <ATen/cpu/vec/vec.h>

#include <iostream>
namespace torch {
namespace aot_inductor {

TEST(TestVec, TestAdd) {
  using Vec = at::vec::Vectorized<int>;
  std::vector<int> a(1024, 1);
  std::vector<int> b(1024, 2);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec actual_vec = a_vec + b_vec;
  std::vector<int> expected(1024, 3);
  Vec expected_vec = Vec::loadu(expected.data());

  for (int i = 0; i < Vec::size(); i++) {
    EXPECT_EQ(expected_vec[i], actual_vec[i]);
  }
}

TEST(TestVec, TestMax) {
  using Vec = at::vec::Vectorized<int>;
  std::vector<int> a(1024, -1);
  std::vector<int> b(1024, 2);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec actual_vec = at::vec::maximum(a_vec, b_vec);
  Vec expected_vec = b_vec;

  for (int i = 0; i < Vec::size(); i++) {
    EXPECT_EQ(expected_vec[i], actual_vec[i]);
  }
}

TEST(TestVec, TestMin) {
  using Vec = at::vec::Vectorized<int>;
  std::vector<int> a(1024, -1);
  std::vector<int> b(1024, 2);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec actual_vec = at::vec::minimum(a_vec, b_vec);
  Vec expected_vec = a_vec;

  for (int i = 0; i < Vec::size(); i++) {
    EXPECT_EQ(expected_vec[i], actual_vec[i]);
  }
}

TEST(TestVec, TestConvert) {
  std::vector<int> a(1024, -1);
  std::vector<float> b(1024, -1.0);
  at::vec::Vectorized<int> a_vec = at::vec::Vectorized<int>::loadu(a.data());
  at::vec::Vectorized<float> b_vec =
      at::vec::Vectorized<float>::loadu(b.data());
  auto actual_vec = at::vec::convert<float>(a_vec);
  auto expected_vec = b_vec;

  for (int i = 0; i < at::vec::Vectorized<int>::size(); i++) {
    EXPECT_EQ(expected_vec[i], actual_vec[i]);
  }
}

TEST(TestVec, TestClampMin) {
  using Vec = at::vec::Vectorized<float>;
  std::vector<float> a(1024, -2.0);
  std::vector<float> min(1024, -1.0);
  Vec a_vec = Vec::loadu(a.data());
  Vec min_vec = Vec::loadu(min.data());
  Vec actual_vec = at::vec::clamp_min(a_vec, min_vec);
  Vec expected_vec = min_vec;

  for (int i = 0; i < Vec::size(); i++) {
    EXPECT_EQ(expected_vec[i], actual_vec[i]);
  }
}

} // namespace aot_inductor
} // namespace torch
