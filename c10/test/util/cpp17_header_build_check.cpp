// Compile-only test to verify that c10 headers mirrored
// to ExecuTorch build with C++17.

#include <gtest/gtest.h>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/bit_cast.h>
#include <c10/util/complex.h>
#include <c10/util/floating_point_utils.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/overflows.h>
#include <c10/util/safe_numerics.h>

#include <torch/headeronly/macros/Export.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/TypeSafeSignMath.h>
#include <torch/headeronly/util/bit_cast.h>
#include <torch/headeronly/util/complex.h>
#include <torch/headeronly/util/floating_point_utils.h>

TEST(Cpp17HeaderBuildCheckTest, HeadersCompile) {
  EXPECT_TRUE(true);
}
