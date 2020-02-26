#include <test/cpp/tensorexpr/tests.h>

#include <gtest/gtest.h>

namespace torch {
namespace jit {

#define TENSOREXPR_GTEST(name) \
  TEST(TensorExprTest, name) { \
    test##name();       \
  }
TH_FORALL_TESTS(TENSOREXPR_GTEST)
#undef TENSOREXPR_GTEST

} // namespace jit
} // namespace torch
