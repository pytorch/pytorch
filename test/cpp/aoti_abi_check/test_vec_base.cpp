#include <gtest/gtest.h>

#include <torch/headeronly/cpu/vec/vec_base.h>

#include <array>
#include <bit>

namespace torch {
namespace aot_inductor {

template <typename T>
void ExpectVecEqual(
    const torch::headeronly::vec::Vectorized<T>& expected,
    const torch::headeronly::vec::Vectorized<T>& actual) {
  using Vec = torch::headeronly::vec::Vectorized<T>;
  // Have to use std::vector for comparison because Vectorized doesn't
  // support operator[] on aarch64
  std::vector<T> expected_data(Vec::size());
  std::vector<T> actual_data(Vec::size());

  expected.store(expected_data.data());
  actual.store(actual_data.data());

  for (int i = 0; i < Vec::size(); i++) {
    EXPECT_EQ(expected_data[i], actual_data[i]);
  }
}

TEST(TestVecBase, TestAdd) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  std::vector<int> a(1024, 1);
  std::vector<int> b(1024, 2);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec actual_vec = a_vec + b_vec;
  std::vector<int> expected(1024, 3);
  Vec expected_vec = Vec::loadu(expected.data());

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestMax) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  std::vector<int> a(1024, -1);
  std::vector<int> b(1024, 2);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec actual_vec = torch::headeronly::vec::maximum(a_vec, b_vec);
  Vec expected_vec = b_vec;

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestMin) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  std::vector<int> a(1024, -1);
  std::vector<int> b(1024, 2);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec actual_vec = torch::headeronly::vec::minimum(a_vec, b_vec);
  Vec expected_vec = a_vec;

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestConvert) {
  std::vector<int> a(1024, -1);
  std::vector<float> b(1024, -1.0);
  torch::headeronly::vec::Vectorized<int> a_vec =
      torch::headeronly::vec::Vectorized<int>::loadu(a.data());
  torch::headeronly::vec::Vectorized<float> b_vec =
      torch::headeronly::vec::Vectorized<float>::loadu(b.data());
  auto actual_vec = torch::headeronly::vec::convert<float>(a_vec);
  auto expected_vec = b_vec;

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestClampMin) {
  using Vec = torch::headeronly::vec::Vectorized<float>;
  std::vector<float> a(1024, -2.0);
  std::vector<float> min(1024, -1.0);
  Vec a_vec = Vec::loadu(a.data());
  Vec min_vec = Vec::loadu(min.data());
  Vec actual_vec = torch::headeronly::vec::clamp_min(a_vec, min_vec);
  Vec expected_vec = min_vec;

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestClampMax) {
  using Vec = torch::headeronly::vec::Vectorized<float>;
  std::vector<float> a(1024, 2.0);
  std::vector<float> max(1024, 1.0);
  Vec a_vec = Vec::loadu(a.data());
  Vec max_vec = Vec::loadu(max.data());
  Vec actual_vec = torch::headeronly::vec::clamp_max(a_vec, max_vec);
  Vec expected_vec = max_vec;

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestClamp) {
  using Vec = torch::headeronly::vec::Vectorized<float>;
  constexpr float input_pattern[] = {-2.0, 0.0, 2.0};
  constexpr float expected_pattern[] = {-1.0, 0.0, 1.0};
  std::vector<float> input(Vec::size());
  std::vector<float> expected(Vec::size());
  for (int i = 0; i < Vec::size(); ++i) {
    input[i] = input_pattern[i % 3];
    expected[i] = expected_pattern[i % 3];
  }

  Vec a_vec = Vec::loadu(input.data());
  Vec min_vec(-1.0);
  Vec max_vec(1.0);
  Vec actual_vec = torch::headeronly::vec::clamp(a_vec, min_vec, max_vec);
  Vec expected_vec = Vec::loadu(expected.data());

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestCast) {
  using FloatVec = torch::headeronly::vec::Vectorized<float>;
  using IntVec = torch::headeronly::vec::Vectorized<int32_t>;
  IntVec actual_vec = torch::headeronly::vec::cast<int32_t>(FloatVec(1.0));
  IntVec expected_vec(std::bit_cast<int32_t>(1.0f));

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestFlip) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  // Fill the lanes with values from 0 through Vec::size() - 1.
  Vec input_vec = Vec::arange(0, 1);
  Vec actual_vec = torch::headeronly::vec::flip(input_vec);
  Vec expected_vec = Vec::arange(Vec::size() - 1, -1);

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestTransposeMxn) {
  std::array<int, 6> input = {1, 2, 3, 4, 5, 6};
  std::array<int, 6> actual = {};
  std::array<int, 6> expected = {1, 4, 2, 5, 3, 6};

  torch::headeronly::vec::transpose_mxn<int, 2, 3>(
      input.data(), 3, actual.data(), 2);

  EXPECT_EQ(expected, actual);
}

TEST(TestVecBase, TestVectorizedMethods) {
  using Vec = torch::headeronly::vec::Vectorized<float>;
  Vec actual_vec = Vec(0.5f).exp2().ceil();
  Vec expected_vec(2.0f);

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestTraits) {
  static_assert(torch::headeronly::vec::is_floating_point<float>::value);
  static_assert(torch::headeronly::vec::is_floating_point_v<c10::Half>);
  static_assert(
      torch::headeronly::vec::is_reduced_floating_point<c10::BFloat16>::value);
  static_assert(
      torch::headeronly::vec::is_reduced_floating_point_v<c10::Half>);
  static_assert(
      torch::headeronly::vec::is_vec_specialized_for<int32_t>::value ==
      torch::headeronly::vec::is_vec_specialized_for_v<int32_t>);
}

} // namespace aot_inductor
} // namespace torch
