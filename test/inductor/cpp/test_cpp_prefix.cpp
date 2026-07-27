#include "../../torchinductor/codegen/cpp_prefix.h"
#include <array>
#include <gtest/gtest.h>

TEST(testCppPrefix, testAtomicAddInt) {
  int x = 0;
  atomic_add(&x, 100);
  EXPECT_EQ(x, 100);
}

TEST(testCppPrefix, testAtomicAddFloat) {
  float x = 0.0f;
  atomic_add(&x, 100.0f);
  EXPECT_EQ(x, 100.0f);
}

TEST(testCppPrefix, testAtomicAddI64) {
  int64_t x = 0.0;
  int64_t y = 100.0;
  atomic_add(&x, y);
  EXPECT_EQ(x, 100);
}

TEST(testCppPrefix, testAtomicAddBool) {
  bool x = false;
  atomic_add(&x, true);
  EXPECT_TRUE(x);

  atomic_add(&x, false);
  EXPECT_TRUE(x);
}

#if INDUCTOR_USE_VECTOR_TYPES()
TEST(testCppPrefix, testAtomicAddVecBoolMask) {
  constexpr int NI = 2;
  constexpr int NV = 1;
  constexpr int len = at::vec::VectorizedN<int64_t, NI>::size();
  static_assert(len >= 4);

  std::array<bool, 4> output = {false, false, true, false};

  __at_align__ std::array<int64_t, len> indices{};
  __at_align__ std::array<bool, len> mask_values{};
  indices[0] = 0;
  mask_values[0] = true;
  indices[1] = 1;
  mask_values[1] = false;
  indices[2] = 2;
  mask_values[2] = true;
  indices[3] = 3;
  mask_values[3] = true;

  auto index =
      at::vec::VectorizedN<int64_t, NI>::loadu(indices.data(), len);
  auto offset = at::vec::VecMask<float, NV>::from(mask_values.data(), len);

  atomic_add_vec<bool, NI, NV>(output.data(), index, offset, 3);

  EXPECT_TRUE(output[0]);
  EXPECT_FALSE(output[1]);
  EXPECT_TRUE(output[2]);
  EXPECT_FALSE(output[3]);
}

TEST(testCppPrefix, testAtomicAddVecBoolMaskDeducedWidth) {
  constexpr int NI = 2;
  constexpr int NV = 1;
  constexpr int NM = 2;
  constexpr int len = at::vec::VectorizedN<int64_t, NI>::size();
  static_assert(len >= 4);

  std::array<bool, 4> output = {false, false, true, false};

  __at_align__ std::array<int64_t, len> indices{};
  __at_align__ std::array<bool, len> mask_values{};
  indices[0] = 0;
  mask_values[0] = true;
  indices[1] = 1;
  mask_values[1] = false;
  indices[2] = 2;
  mask_values[2] = true;
  indices[3] = 3;
  mask_values[3] = true;

  auto index =
      at::vec::VectorizedN<int64_t, NI>::loadu(indices.data(), len);
  auto offset = at::vec::VecMask<int64_t, NM>::from(mask_values.data(), len);

  atomic_add_vec<bool, NI, NV>(output.data(), index, offset, 3);

  EXPECT_TRUE(output[0]);
  EXPECT_FALSE(output[1]);
  EXPECT_TRUE(output[2]);
  EXPECT_FALSE(output[3]);
}
#endif
