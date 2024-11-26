#include <gtest/gtest.h>
#include <stdexcept>
#include <torch/torch.h>
#include <ATen/native/mps/MetalShaderLibrary.h>

using namespace at::native::mps;
TEST(MPSTestMetalLibrary, ShaderCreation) {
   MetalShaderLibrary lib("// Empty library");
   ASSERT_EQ(lib.getFunctionNames().size(), 0);
}
TEST(MPSTestMetalLibrary, SyntaxErrorThrows) {
  ASSERT_THROW(new DynamicMetalShaderLibrary("printf(x);"), c10::Error);
}
TEST(MPSTestMetalLibrary, ArangeShader) {
  auto y = torch::arange(10.0, at::device(at::kMPS));
  auto x = torch::empty(10, at::device(at::kMPS));
  DynamicMetalShaderLibrary lib(R"MTL(
  kernel void foo(device float* t, uint idx [[thread_position_in_grid]]) {
    t[idx] = idx;
  }
  )MTL");
  auto func = lib.getKernelFunction("foo");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, x);
     func->dispatch(x.numel());
  });
  ASSERT_TRUE((x==y).all().item().toBool());
}
