#include <gtest/gtest.h>

#include <ATen/cpu/vec/vec.h>

namespace torch {
namespace aot_inductor {

template <typename T>
void ExpectVecEqual(
    const at::vec::Vectorized<T>& expected,
    const at::vec::Vectorized<T>& actual) {
  using Vec = at::vec::Vectorized<T>;
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

TEST(TestVec, TestConvert) {
  std::vector<int> a(1024, -1);
  std::vector<float> b(1024, -1.0);
  at::vec::Vectorized<int> a_vec = at::vec::Vectorized<int>::loadu(a.data());
  at::vec::Vectorized<float> b_vec =
      at::vec::Vectorized<float>::loadu(b.data());
  auto actual_vec = at::vec::convert<float>(a_vec);
  auto expected_vec = b_vec;

  ExpectVecEqual(expected_vec, actual_vec);
}

} // namespace aot_inductor
} // namespace torch
