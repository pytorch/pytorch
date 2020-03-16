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

#ifdef TORCH_ENABLE_LLVM
#define TENSOREXPR_GTEST_LLVM(name)   \
  TEST(TensorExprTest, name##_LLVM) { \
    test##name();                     \
  }
TH_FORALL_TESTS_LLVM(TENSOREXPR_GTEST_LLVM)
#undef TENSOREXPR_GTEST_LLVM
#endif

#ifdef USE_CUDA
#define TENSOREXPR_GTEST_CUDA(name)   \
  TEST(TensorExprTest, name##_CUDA) { \
    test##name();                     \
  }
TH_FORALL_TESTS_CUDA(TENSOREXPR_GTEST_CUDA)
#undef TENSOREXPR_GTEST_CUDA
#endif

} // namespace jit
} // namespace torch
