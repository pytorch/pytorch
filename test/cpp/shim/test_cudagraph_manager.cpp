// Checks the AOTI runtime's CudaGraphManager, which generated code compiles
// into the model .so, against the C shim's CUDA graph functions.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_runtime/cudagraph_manager.h>

#include <array>
#include <string>
#include <thread>
#include <vector>

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
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    TORCH_CHECK(
        !fail_in_capture || status == cudaStreamCaptureStatusNone,
        "region can't be captured");
  }

  Compute compute;
  int calls = 0;
  bool fail = false;
  bool fail_in_capture = false;
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

TEST_F(CudaGraphManagerTest, ThreadsRunEagerlyBeforeCapturing) {
  CudaGraphManager manager(0);
  Region region([](const at::Tensor& x) { return x.mm(x); });
  auto x = at::randn({16, 16}, at::kCUDA);
  run(manager, region, x);
  // A new thread has no cuBLAS handle yet, and can't create one in a capture.
  std::thread([&] {
    EXPECT_TRUE(run(manager, region, x).allclose(x.mm(x)));
    EXPECT_EQ(manager.num_graphs(), 0);
    EXPECT_TRUE(run(manager, region, x).allclose(x.mm(x)));
    EXPECT_EQ(manager.num_graphs(), 1);
  }).join();
  EXPECT_EQ(region.calls, 3);
}

TEST_F(CudaGraphManagerTest, ThreadIsWarmOnlyAfterASuccessfulCall) {
  CudaGraphManager manager(0);
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  run(manager, region, x);
  std::thread([&] {
    region.fail = true;
    EXPECT_THROW(run(manager, region, x), c10::Error);
    region.fail = false;
    // The failed call doesn't count, so this one runs eagerly too.
    EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
    EXPECT_EQ(manager.num_graphs(), 0);
    EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
    EXPECT_EQ(manager.num_graphs(), 1);
  }).join();
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

TEST_F(CudaGraphManagerTest, HandlesInputsWithGaps) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto make_input = [] { return at::randn({6, 8}, at::kCUDA).narrow(1, 2, 3); };
  run(manager, region, make_input());

  auto x = make_input();
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
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
  for (int i = 0; i < 2; ++i) {
    AtenTensorHandle outputs[2] = {nullptr, nullptr};
    manager.run(
        region,
        &input,
        1,
        outputs,
        2,
        at::cuda::getCurrentCUDAStream().stream());
    EXPECT_NE(outputs[0], outputs[1]);
    EXPECT_TRUE(tensor_of(outputs[0]).equal(affine(x)));
    EXPECT_TRUE(tensor_of(outputs[1]).is_same(tensor_of(outputs[0])));
    aoti_torch_delete_tensor_object(outputs[0]);
    aoti_torch_delete_tensor_object(outputs[1]);
  }
  EXPECT_EQ(calls, 1);
}

TEST_F(CudaGraphManagerTest, ReplaysAdvanceRng) {
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

TEST_F(CudaGraphManagerTest, ReplayedOutputsDontAliasInputs) {
  // Warm-up calls run the region eagerly, so their outputs can alias inputs.
  CudaGraphManager manager(0);
  Region region([](const at::Tensor& x) { return x.narrow(0, 2, 4); });
  auto x = at::randn({8}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, x).is_alias_of(x));

  auto y = at::randn({8}, at::kCUDA);
  auto y_out = run(manager, region, y);
  EXPECT_EQ(manager.num_graphs(), 1);
  EXPECT_FALSE(y_out.is_alias_of(y));
  EXPECT_TRUE(y_out.equal(y.narrow(0, 2, 4)));
  auto z = at::randn({8}, at::kCUDA);
  auto z_out = run(manager, region, z);
  EXPECT_FALSE(z_out.is_alias_of(z));
  EXPECT_TRUE(z_out.equal(z.narrow(0, 2, 4)));
  EXPECT_TRUE(y_out.equal(y.narrow(0, 2, 4)));
}

TEST_F(CudaGraphManagerTest, HandlesEmptyAndScalarInputs) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  for (auto sizes : {std::vector<int64_t>{0, 4}, std::vector<int64_t>{}}) {
    for (int i = 0; i < 2; ++i) {
      auto x = at::randn(sizes, at::kCUDA);
      EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
    }
  }
  EXPECT_EQ(manager.num_graphs(), 2);
  EXPECT_EQ(region.calls, 2);
}

TEST_F(CudaGraphManagerTest, OrdersReplaysOnDifferentStreams) {
  CudaGraphManager manager(0, no_warmup());
  // Long enough that, unordered, the second call's input copy would land
  // while the first replay still reads the shared input buffer.
  Region region([](const at::Tensor& x) {
    at::cuda::sleep(1 << 28);
    return affine(x);
  });
  auto s1 = at::cuda::getStreamFromPool();
  auto s2 = at::cuda::getStreamFromPool();
  auto run_on = [&](const at::cuda::CUDAStream& stream, const at::Tensor& x) {
    at::cuda::CUDAStreamGuard guard(stream);
    return run(manager, region, x);
  };
  auto a = at::randn({16}, at::kCUDA);
  auto b = at::randn({16}, at::kCUDA);
  c10::cuda::device_synchronize();
  run_on(s1, a);
  auto a_out = run_on(s1, a);
  auto b_out = run_on(s2, b);
  c10::cuda::device_synchronize();
  EXPECT_EQ(region.calls, 1);
  EXPECT_TRUE(a_out.equal(affine(a)));
  EXPECT_TRUE(b_out.equal(affine(b)));
}

TEST_F(CudaGraphManagerTest, CaptureOnlyFailureRunsEagerly) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  region.fail_in_capture = true;
  auto x = at::randn({16}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(region.calls, 2);
  // The key isn't captured again.
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(region.calls, 3);
  EXPECT_EQ(manager.num_graphs(), 0);

  // Other keys still capture.
  region.fail_in_capture = false;
  auto y = at::randn({8}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, y).equal(affine(y)));
  EXPECT_EQ(manager.num_graphs(), 1);
}

TEST_F(CudaGraphManagerTest, ForgetsUncapturableKeysPastLimit) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  region.fail_in_capture = true;
  struct IgnoreWarnings : c10::WarningHandler {
    void process(const c10::Warning& /*warning*/) override {}
  } ignore_warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&ignore_warnings);
  for (size_t n = 1; n <= CudaGraphManager::kMaxWarmupKeys + 1; ++n) {
    run(manager, region, at::randn({static_cast<int64_t>(n)}, at::kCUDA));
  }
  // The first shape was forgotten, so its capture is tried again.
  int calls = region.calls;
  run(manager, region, at::randn({1}, at::kCUDA));
  EXPECT_EQ(region.calls - calls, 2);
  // The last one wasn't, so it just runs eagerly.
  calls = region.calls;
  auto n = static_cast<int64_t>(CudaGraphManager::kMaxWarmupKeys + 1);
  run(manager, region, at::randn({n}, at::kCUDA));
  EXPECT_EQ(region.calls - calls, 1);
}

TEST_F(CudaGraphManagerTest, RejectsADifferentOutputCount) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  run(manager, region, x);
  run(manager, region, x);
  ASSERT_EQ(manager.num_graphs(), 1);
  auto input = reinterpret_cast<AtenTensorHandle>(&x);
  std::array<AtenTensorHandle, 2> outputs{};
  EXPECT_THROW(
      manager.run(
          region,
          &input,
          1,
          outputs.data(),
          outputs.size(),
          at::cuda::getCurrentCUDAStream().stream()),
      std::runtime_error);
}

TEST_F(CudaGraphManagerTest, WarnsOnceWhenCaptureFails) {
  struct CollectWarnings : c10::WarningHandler {
    std::vector<std::string> messages;
    void process(const c10::Warning& warning) override {
      messages.push_back(warning.msg());
    }
  } warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  region.fail_in_capture = true;
  auto x = at::randn({16}, at::kCUDA);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  }
  ASSERT_EQ(warnings.messages.size(), 1);
  EXPECT_NE(
      warnings.messages[0].find("region can't be captured"), std::string::npos);
}

TEST_F(CudaGraphManagerTest, RegionErrorAlsoFailingEagerlyIsRethrown) {
  // The error isn't about capturing, so the key is captured once the region
  // works.
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

TEST_F(CudaGraphManagerTest, RegionErrorDuringCaptureKeepsOtherGraphs) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto a = at::randn({8}, at::kCUDA);
  auto b = at::randn({4}, at::kCUDA);
  run(manager, region, a);
  region.fail = true;
  EXPECT_THROW(run(manager, region, b), c10::Error);
  EXPECT_EQ(manager.num_graphs(), 1);

  region.fail = false;
  EXPECT_TRUE(run(manager, region, b).equal(affine(b)));
  EXPECT_EQ(manager.num_graphs(), 2);
  a = at::randn({8}, at::kCUDA);
  b = at::randn({4}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, a).equal(affine(a)));
  EXPECT_TRUE(run(manager, region, b).equal(affine(b)));
  // The failing call ran the region twice: in the capture and eagerly.
  EXPECT_EQ(region.calls, 4);
}

TEST_F(CudaGraphManagerTest, RejectsInputsOnOtherDevices) {
  CudaGraphManager manager(0);
  Region region(affine);
  // Even during warm-up, which would otherwise run the region eagerly.
  EXPECT_THROW(run(manager, region, at::randn({8})), std::runtime_error);
  if (c10::cuda::device_count() >= 2) {
    EXPECT_THROW(
        run(manager, region, at::randn({8}, at::Device(at::kCUDA, 1))),
        std::runtime_error);
  }
  EXPECT_EQ(region.calls, 0);
}

TEST_F(CudaGraphManagerTest, KeysOnAllInputs) {
  CudaGraphManager manager(0, no_warmup());
  int calls = 0;
  auto region = [&](AtenTensorHandle* inputs,
                    AtenTensorHandle* outputs,
                    cudaStream_t stream) {
    calls++;
    const at::Tensor& x = tensor_of(inputs[0]);
    at::cuda::CUDAStreamGuard guard(
        at::cuda::getStreamFromExternal(stream, x.get_device()));
    outputs[0] = new_handle(x + tensor_of(inputs[1]).sum());
  };
  auto run_region = [&](at::Tensor x, at::Tensor y) {
    AtenTensorHandle inputs[2] = {
        reinterpret_cast<AtenTensorHandle>(&x),
        reinterpret_cast<AtenTensorHandle>(&y)};
    AtenTensorHandle output = nullptr;
    manager.run(
        region,
        inputs,
        2,
        &output,
        1,
        at::cuda::getCurrentCUDAStream().stream());
    at::Tensor result = tensor_of(output);
    aoti_torch_delete_tensor_object(output);
    return result;
  };
  auto x = at::randn({8}, at::kCUDA);
  run_region(x, at::randn({4}, at::kCUDA));
  run_region(x, at::randn({2}, at::kCUDA));
  EXPECT_EQ(manager.num_graphs(), 2);

  x = at::randn({8}, at::kCUDA);
  auto y = at::randn({4}, at::kCUDA);
  EXPECT_TRUE(run_region(x, y).allclose(x + y.sum()));
  EXPECT_EQ(calls, 2);
}

TEST_F(CudaGraphManagerTest, ReturnsNullOutputs) {
  CudaGraphManager manager(0, no_warmup());
  int calls = 0;
  auto region = [&](AtenTensorHandle* inputs,
                    AtenTensorHandle* outputs,
                    cudaStream_t stream) {
    calls++;
    const at::Tensor& x = tensor_of(inputs[0]);
    at::cuda::CUDAStreamGuard guard(
        at::cuda::getStreamFromExternal(stream, x.get_device()));
    outputs[0] = nullptr;
    outputs[1] = new_handle(affine(x));
  };
  auto x = at::randn({16}, at::kCUDA);
  auto input = reinterpret_cast<AtenTensorHandle>(&x);
  for (int i = 0; i < 2; ++i) {
    AtenTensorHandle outputs[2] = {nullptr, nullptr};
    manager.run(
        region,
        &input,
        1,
        outputs,
        2,
        at::cuda::getCurrentCUDAStream().stream());
    EXPECT_EQ(outputs[0], nullptr);
    EXPECT_TRUE(tensor_of(outputs[1]).equal(affine(x)));
    aoti_torch_delete_tensor_object(outputs[1]);
  }
  EXPECT_EQ(calls, 1);
}

TEST_F(CudaGraphManagerTest, MaxGraphsZeroNeverCaptures) {
  auto options = no_warmup();
  options.max_graphs = 0;
  CudaGraphManager manager(0, options);
  Region region(affine);
  auto x = at::randn({8}, at::kCUDA);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  }
  EXPECT_EQ(region.calls, 3);
  EXPECT_EQ(manager.num_graphs(), 0);
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
  // The key warms up again, then is captured again.
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
