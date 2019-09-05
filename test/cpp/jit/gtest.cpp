#include <test/cpp/jit/tests.h>

#include <gtest/gtest.h>

namespace torch {
namespace jit {

#define JIT_GTEST(name) \
  TEST(JitTest, name) { \
    test##name();       \
  }
TH_FORALL_TESTS(JIT_GTEST)
#undef JIT_TEST

#define JIT_GTEST_CUDA(name)   \
  TEST(JitTest, name##_CUDA) { \
    test##name();              \
  }
TH_FORALL_TESTS_CUDA(JIT_GTEST_CUDA)
#undef JIT_TEST_CUDA

} // namespace jit
} // namespace torch
