#include <gtest/gtest.h>
#define C10_DEFINE_TEST(a, b) TEST(a, b)
#define C10_ASSERT_NEAR(a, b, tol) ASSERT_NEAR(a, b, tol)
#include <c10/test/util/complex_math_test_common.h>
