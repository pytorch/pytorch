#include <gtest/gtest.h>
#include <stdexcept>
#include <torch/torch.h>
#include <ATen/native/mps/MetalShaderLibrary.h>

using namespace at::native::mps;
TEST(MPSTestMetalLibrary, ShaderCreation) {
   DynamicMetalShaderLibrary lib("// Empty library");
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

TEST(MPSTestMetalLibrary, ArangeWithArgsShader) {
  const auto size = 10;
  const float start = .25;
  const float step = .4;
  auto x = torch::empty(size, at::device(at::kMPS));
  auto y = torch::arange(start, start + size * step, step, at::device(at::kMPS));
  ASSERT_EQ(x.numel(), y.numel());
  DynamicMetalShaderLibrary lib(R"MTL(
  kernel void foo(device float* t, constant float& start, constant float& step, uint idx [[thread_position_in_grid]]) {
    t[idx] = start + idx * step;
  }
  )MTL");
  auto func = lib.getKernelFunction("foo");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, x);
     func->setArg(1, start);
     func->setArg(2, step);
     func->dispatch(x.numel());
  });
  ASSERT_TRUE((x==y).all().item().toBool());
}
TEST(MPSTestMetalLibrary, Arange2DShader) {
  const auto size = 16;
  auto x = torch::empty({size, size}, at::device(at::kMPS));
  DynamicMetalShaderLibrary lib(R"MTL(
  kernel void full(device float* t, constant ulong2& strides, uint2 idx [[thread_position_in_grid]]) {
    t[idx.x*strides.x + idx.y*strides.y] = idx.x + 33.0 * idx.y;
  }
  )MTL");
  auto func = lib.getKernelFunction("full");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, x);
     func->setArg(1, x.strides());
     func->dispatch({static_cast<uint64_t>(x.size(0)), static_cast<uint64_t>(x.size(1))});
  });
  ASSERT_EQ(x.sum().item().to<int>(), 65280);
}
