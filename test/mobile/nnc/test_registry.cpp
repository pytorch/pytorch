#include <gtest/gtest.h>
#include <torch/csrc/jit/mobile/nnc/registry.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

extern "C" {
int generated_asm_kernel_foo(void**) {
  return 1;
}

int generated_asm_kernel_bar(void**) {
  return 2;
}
} // extern "C"

REGISTER_NNC_KERNEL("foo:v1:VERTOKEN", generated_asm_kernel_foo)
REGISTER_NNC_KERNEL("bar:v1:VERTOKEN", generated_asm_kernel_bar)

TEST(MobileNNCRegistryTest, FindAndRun) {
  auto foo_kernel = registry::get_nnc_kernel("foo:v1:VERTOKEN");
  EXPECT_EQ(foo_kernel->execute(nullptr), 1);

  auto bar_kernel = registry::get_nnc_kernel("bar:v1:VERTOKEN");
  EXPECT_EQ(bar_kernel->execute(nullptr), 2);
}

TEST(MobileNNCRegistryTest, NoKernel) {
  EXPECT_EQ(registry::has_nnc_kernel("missing"), false);
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
