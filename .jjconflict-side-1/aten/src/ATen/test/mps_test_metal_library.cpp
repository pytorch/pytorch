#include <gtest/gtest.h>
#include <stdexcept>
#include <torch/torch.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSProfiler.h>
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

TEST(MPSTestMetalLibrary, ArgumentBuffers) {
  constexpr auto nbuffers = 64;
  const auto size = 32;
  std::vector<at::Tensor> ibuffers;
  std::vector<void *> ibuffers_gpu_ptrs;
  for([[maybe_unused]] auto idx: c10::irange(nbuffers)) {
    ibuffers.push_back(torch::rand({size}, at::device(at::kMPS)));
    ibuffers_gpu_ptrs.push_back(get_tensor_gpu_address(ibuffers.back()));
  }
  auto output = torch::empty({size}, at::device(at::kMPS));
  DynamicMetalShaderLibrary lib(R"MTL(
  constant constexpr auto nbuffers = 64;
  struct Inputs {
    metal::array<device float *, nbuffers> args;
  };

  kernel void sum_all(device float* output, constant Inputs& inputs, uint idx [[thread_position_in_grid]]) {
    output[idx] = 0;
    for(auto i = 0; i < nbuffers; ++i) {
      output[idx] += inputs.args[i][idx];
    }
  }
  )MTL");
  auto func = lib.getKernelFunction("sum_all");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, output);
     func->setArg(1, ibuffers_gpu_ptrs);
     func->dispatch(size);
  });
  // Compute sum of all 64 input tensors
  auto result = torch::zeros({size}, at::device(at::kMPS));
  for(auto buf: ibuffers) {
    result += buf;
  }
  ASSERT_EQ(result.sum().item().to<float>(), output.sum().item().to<float>());
}
