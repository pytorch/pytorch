#include <gtest/gtest.h>

#include <torch/headeronly/util/jiterator_macros.h>

jiterator_also_stringify_as(
    jiterator_code(template <typename T> JITERATOR_HOST_DEVICE T
                       dot_product_4(const T* lhs, const T* rhs) {
                         T result = 0;
                         for (int index = 0; index < 4; ++index) {
                           result += lhs[index] * rhs[index];
                         }
                         return result;
                       }),
    dot_product_4_string)

TEST(TestJiteratorMacros, DefinesCode) {
  const float lhs[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float rhs[] = {5.0f, 6.0f, 7.0f, 8.0f};

  EXPECT_FLOAT_EQ(dot_product_4(lhs, rhs), 70.0f);
}
