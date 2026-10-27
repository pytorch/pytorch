// Checks the AOTI runtime's CudaGraphManager, which generated code compiles
// into the model .so, against the C shim's CUDA graph functions.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_runtime/cudagraph_manager.h>

using torch::aot_inductor::CudaGraphManager;
using torch::aot_inductor::CudaGraphOptions;

namespace {

AtenTensorHandle new_handle(at::Tensor tensor) {
  return reinterpret_cast<AtenTensorHandle>(new at::Tensor(std::move(tensor)));
}

const at::Tensor& tensor_of(AtenTensorHandle handle) {
  return *reinterpret_cast<at::Tensor*>(handle);
}

using Compute = at::Tensor (*)(const at::Tensor&);

// A region computing one output from one input with ATen, counting its calls.
struct Region {
  explicit Region(Compute compute) : compute(compute) {}

  void operator()(
      AtenTensorHandle* inputs,
      AtenTensorHandle* outputs,
      cudaStream_t stream) {
    calls++;
    const at::Tensor& x = tensor_of(inputs[0]);
    at::cuda::CUDAStreamGuard guard(
        at::cuda::getStreamFromExternal(stream, x.get_device()));
    outputs[0] = new_handle(compute(x));
    // A failing region may already have launched work and written outputs.
    TORCH_CHECK(!fail, "region failed");
  }

  Compute compute;
  int calls = 0;
  bool fail = false;
};

at::Tensor affine(const at::Tensor& x) {
  return x * 2 + 1;
}

at::Tensor run(CudaGraphManager& manager, Region& region, at::Tensor x) {
  auto input = reinterpret_cast<AtenTensorHandle>(&x);
  AtenTensorHandle output = nullptr;
  manager.run(
      region, &input, 1, &output, 1, at::cuda::getCurrentCUDAStream().stream());
  at::Tensor result = tensor_of(output);
  aoti_torch_delete_tensor_object(output);
  return result;
}

CudaGraphOptions no_warmup() {
  CudaGraphOptions options;
  options.warmup_calls = 0;
  return options;
}

class CudaGraphManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::cuda::device_count() == 0) {
      GTEST_SKIP() << "CUDA is not available";
    }
  }
};

} // namespace

TEST_F(CudaGraphManagerTest, WarmsUpThenCapturesThenReplays) {
  CudaGraphManager manager(0);
  Region region(affine);

  auto x = at::randn({16}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(region.calls, 1);
  EXPECT_EQ(manager.num_graphs(), 0);

  auto y = at::randn({16}, at::kCUDA);
  auto y_out = run(manager, region, y);
  EXPECT_TRUE(y_out.equal(affine(y)));
  EXPECT_EQ(region.calls, 2);
  EXPECT_EQ(manager.num_graphs(), 1);

  auto z = at::randn({16}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, z).equal(affine(z)));
  EXPECT_EQ(region.calls, 2);
  // Outputs are clones, so a replay doesn't overwrite earlier results.
  EXPECT_TRUE(y_out.equal(affine(y)));
}

TEST_F(CudaGraphManagerTest, KeysOnShapes) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto a = at::randn({8}, at::kCUDA);
  auto b = at::randn({4, 2}, at::kCUDA);
  run(manager, region, a);
  run(manager, region, b);
  EXPECT_EQ(manager.num_graphs(), 2);

  EXPECT_TRUE(run(manager, region, a).equal(affine(a)));
  EXPECT_TRUE(run(manager, region, b).equal(affine(b)));
  EXPECT_EQ(region.calls, 2);
}

TEST_F(CudaGraphManagerTest, PreservesLayouts) {
  CudaGraphManager manager(0, no_warmup());
  // A strided view with a storage offset, so the output is not contiguous.
  Region region([](const at::Tensor& x) { return (x * 2).narrow(0, 1, 2); });
  auto make_input = [] { return at::randn({6, 4}, at::kCUDA).t(); };
  run(manager, region, make_input());

  auto x = make_input();
  auto out = run(manager, region, x);
  auto expected = (x * 2).narrow(0, 1, 2);
  EXPECT_TRUE(out.equal(expected));
  EXPECT_EQ(out.strides(), expected.strides());
  EXPECT_EQ(region.calls, 1);
}

TEST_F(CudaGraphManagerTest, HandlesOverlappingInputs) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto make_input = [] { return at::randn({1, 6}, at::kCUDA).expand({4, 6}); };
  run(manager, region, make_input());

  auto x = make_input();
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(region.calls, 1);
}

TEST_F(CudaGraphManagerTest, ReturnsDuplicateOutputs) {
  CudaGraphManager manager(0, no_warmup());
  int calls = 0;
  auto region = [&](AtenTensorHandle* inputs,
                    AtenTensorHandle* outputs,
                    cudaStream_t stream) {
    calls++;
    const at::Tensor& x = tensor_of(inputs[0]);
    at::cuda::CUDAStreamGuard guard(
        at::cuda::getStreamFromExternal(stream, x.get_device()));
    outputs[0] = new_handle(affine(x));
    outputs[1] = outputs[0];
  };
  auto x = at::randn({16}, at::kCUDA);
  auto input = reinterpret_cast<AtenTensorHandle>(&x);
  AtenTensorHandle outputs[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; ++i) {
    manager.run(
        region,
        &input,
        1,
        outputs,
        2,
        at::cuda::getCurrentCUDAStream().stream());
  }
  EXPECT_EQ(calls, 1);
  EXPECT_NE(outputs[0], outputs[1]);
  EXPECT_TRUE(tensor_of(outputs[0]).equal(affine(x)));
  EXPECT_TRUE(tensor_of(outputs[1]).is_same(tensor_of(outputs[0])));
  aoti_torch_delete_tensor_object(outputs[0]);
  aoti_torch_delete_tensor_object(outputs[1]);
}

TEST_F(CudaGraphManagerTest, ReplaysAdvanceRng) {
  // Without a warm-up, the default generator would be created during the
  // capture, which isn't allowed.
  at::cuda::detail::getDefaultCUDAGenerator();
  CudaGraphManager manager(0, no_warmup());
  Region region([](const at::Tensor& x) { return at::rand_like(x); });
  auto x = at::empty({1024}, at::kCUDA);
  auto first = run(manager, region, x);
  auto second = run(manager, region, x);
  EXPECT_EQ(region.calls, 1);
  EXPECT_FALSE(first.equal(second));
}

TEST_F(CudaGraphManagerTest, CallerCaptureRunsRegionDirectly) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  at::Tensor out;
  at::cuda::CUDAGraph graph;
  {
    at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromPool());
    graph.capture_begin();
    out = run(manager, region, x);
    graph.capture_end();
  }
  EXPECT_EQ(region.calls, 1);
  EXPECT_EQ(manager.num_graphs(), 0);

  x.copy_(at::randn({16}, at::kCUDA));
  graph.replay();
  EXPECT_TRUE(out.equal(affine(x)));
}

TEST_F(CudaGraphManagerTest, RegionErrorDuringCaptureIsRethrown) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  region.fail = true;
  auto x = at::randn({16}, at::kCUDA);
  EXPECT_THROW(run(manager, region, x), c10::Error);
  EXPECT_EQ(manager.num_graphs(), 0);

  region.fail = false;
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(manager.num_graphs(), 1);
}

TEST_F(CudaGraphManagerTest, MaxGraphsRunsNewKeysEagerly) {
  auto options = no_warmup();
  options.max_graphs = 1;
  CudaGraphManager manager(0, options);
  Region region(affine);
  auto a = at::randn({8}, at::kCUDA);
  auto b = at::randn({4}, at::kCUDA);
  run(manager, region, a);
  EXPECT_TRUE(run(manager, region, b).equal(affine(b)));
  EXPECT_TRUE(run(manager, region, b).equal(affine(b)));
  EXPECT_EQ(region.calls, 3);
  EXPECT_EQ(manager.num_graphs(), 1);

  EXPECT_TRUE(run(manager, region, a).equal(affine(a)));
  EXPECT_EQ(region.calls, 3);
}

TEST_F(CudaGraphManagerTest, RejectsNegativeOptions) {
  CudaGraphOptions negative_warmups;
  negative_warmups.warmup_calls = -1;
  EXPECT_THROW(CudaGraphManager(0, negative_warmups), std::runtime_error);

  CudaGraphOptions negative_max_graphs;
  negative_max_graphs.max_graphs = -1;
  EXPECT_THROW(CudaGraphManager(0, negative_max_graphs), std::runtime_error);
}

TEST_F(CudaGraphManagerTest, ResetDropsGraphs) {
  CudaGraphManager manager(0);
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  run(manager, region, x);
  run(manager, region, x);
  EXPECT_EQ(manager.num_graphs(), 1);

  manager.reset();
  EXPECT_EQ(manager.num_graphs(), 0);
  // The key warms up again, then is captured into a new pool.
  run(manager, region, x);
  EXPECT_EQ(region.calls, 3);
  EXPECT_EQ(manager.num_graphs(), 0);
  auto y = at::randn({16}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, y).equal(affine(y)));
  EXPECT_EQ(region.calls, 4);
  EXPECT_EQ(manager.num_graphs(), 1);
}

TEST_F(CudaGraphManagerTest, ForgetsWarmupsPastLimit) {
  CudaGraphManager manager(0);
  Region region(affine);
  // Fill warm-up tracking with one-off shapes, then add one more.
  for (size_t n = 1; n <= CudaGraphManager::kMaxWarmupKeys + 1; ++n) {
    run(manager, region, at::randn({static_cast<int64_t>(n)}, at::kCUDA));
  }
  // The first shape was forgotten, so it warms up again before capture.
  auto first = at::randn({1}, at::kCUDA);
  run(manager, region, first);
  EXPECT_EQ(manager.num_graphs(), 0);
  run(manager, region, first);
  EXPECT_EQ(manager.num_graphs(), 1);
}
