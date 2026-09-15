#include <gtest/gtest.h>

#include <ATen/cpu/vec/vec.h>
#include <torch/headeronly/cpu/vec/vec_base.h>

#include <vector>

namespace torch {
namespace aot_inductor {

template <typename T>
void ExpectVecEqual(
    const torch::headeronly::vec::Vectorized<T>& expected,
    const torch::headeronly::vec::Vectorized<T>& actual) {
  using Vec = torch::headeronly::vec::Vectorized<T>;
  // Have to use std::vector for comparison because at::vec::Vectorized doesn't
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

TEST(TestVecBase, TestClamp) {
  using Vec = torch::headeronly::vec::Vectorized<float>;
  std::vector<float> a(1024, 2.0);
  std::vector<float> min(1024, -1.0);
  std::vector<float> max(1024, 1.0);
  Vec a_vec = Vec::loadu(a.data());
  Vec min_vec = Vec::loadu(min.data());
  Vec max_vec = Vec::loadu(max.data());
  Vec actual_vec = torch::headeronly::vec::clamp(a_vec, min_vec, max_vec);
  Vec expected_vec = max_vec;

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

TEST(TestVecBase, TestFmadd) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  std::vector<int> a(1024, 2);
  std::vector<int> b(1024, 3);
  std::vector<int> c(1024, 4);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec c_vec = Vec::loadu(c.data());
  Vec actual_vec = torch::headeronly::vec::fmadd(a_vec, b_vec, c_vec);
  std::vector<int> expected(1024, 10);
  Vec expected_vec = Vec::loadu(expected.data());

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestFnmadd) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  std::vector<int> a(1024, 2);
  std::vector<int> b(1024, 3);
  std::vector<int> c(1024, 4);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec c_vec = Vec::loadu(c.data());
  Vec actual_vec = torch::headeronly::vec::fnmadd(a_vec, b_vec, c_vec);
  std::vector<int> expected(1024, -2);
  Vec expected_vec = Vec::loadu(expected.data());

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestFmsub) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  std::vector<int> a(1024, 2);
  std::vector<int> b(1024, 3);
  std::vector<int> c(1024, 4);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec c_vec = Vec::loadu(c.data());
  Vec actual_vec = torch::headeronly::vec::fmsub(a_vec, b_vec, c_vec);
  std::vector<int> expected(1024, 2);
  Vec expected_vec = Vec::loadu(expected.data());

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestFnmsub) {
  using Vec = torch::headeronly::vec::Vectorized<int>;
  std::vector<int> a(1024, 2);
  std::vector<int> b(1024, 3);
  std::vector<int> c(1024, 4);
  Vec a_vec = Vec::loadu(a.data());
  Vec b_vec = Vec::loadu(b.data());
  Vec c_vec = Vec::loadu(c.data());
  Vec actual_vec = torch::headeronly::vec::fnmsub(a_vec, b_vec, c_vec);
  std::vector<int> expected(1024, -10);
  Vec expected_vec = Vec::loadu(expected.data());

  ExpectVecEqual(expected_vec, actual_vec);
}

TEST(TestVecBase, TestSpecializationTraits) {
  EXPECT_EQ(
      torch::headeronly::vec::is_vec_specialized_for<int>::value,
      torch::headeronly::vec::is_vec_specialized_for_v<int>);
}

} // namespace aot_inductor
} // namespace torch
