#include <ATen/Operators.h>
#include <ATen/test/test_assert.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>

#include <cassert>

using namespace at;

TEST(PackedtensoraccessorTest, TransposeTest) {
  manual_seed(123);
  /* test a 3d tensor */
  constexpr int dimension = 3;
  constexpr std::array<long, dimension> sizes{3, 4, 5};
  Tensor t = rand(sizes, CPU(kFloat));
  auto original = t.packed_accessor64<float, dimension, DefaultPtrTraits>();
  auto transposed = original.transpose(0, 2);
  ASSERT_EQ(original.size(0), transposed.size(2));
  ASSERT_EQ(original.size(1), transposed.size(1));
  ASSERT_EQ(original.size(2), transposed.size(0));
  for (int i = 0; i < sizes[0]; i++)
    for (int j = 0; j < sizes[1]; j++)
      for (int k = 0; k < sizes[2]; k++)
        ASSERT_EQ(original[i][j][k], transposed[k][j][i]);

  /* test the special case of a 1d tensor */
  int size = 3;
  t = rand({size}, CPU(kFloat));
  auto original_1d = t.packed_accessor64<float, 1, DefaultPtrTraits>();
  auto transposed_1d = original_1d.transpose(0, 0);
  for (int i = 0; i < size; i++)
    ASSERT_EQ(original_1d[i], transposed_1d[i]);

  /* test the error conditions */
  ASSERT_THROW(original.transpose(2, 5), c10::IndexError);
  ASSERT_THROW(original_1d.transpose(1, 0), c10::IndexError);
}

TEST(PackedtensoraccessorTest, ViewTest) {
  manual_seed(123);
  constexpr int dimension = 3;
  constexpr std::array<long, dimension> sizes{3, 4, 5};
  Tensor t = rand(sizes, CPU(kFloat));
  auto view_3d = t.packed_accessor64<float, dimension, DefaultPtrTraits>();
  auto view_2d = view_3d.view<2>();
  for (int i = 0; i < sizes[1]; i++)
    for (int j = 0; j < sizes[2]; j++)
      ASSERT_EQ(view_3d[0][i][j], view_2d[i][j]);

  auto view_1d = view_3d.view<1>();
  for (int i = 0; i < sizes[2]; i++)
    ASSERT_EQ(view_2d[0][i], view_1d[i]);

  /* now view the 1d data at the last index of of view_2d */
  auto view_1d_end = view_2d.view<1>(view_2d[sizes[1] - 1].data());
  for (int i = 0; i < sizes[2]; i++)
    ASSERT_EQ(view_2d[sizes[1] - 1][i], view_1d_end[i]);
}
