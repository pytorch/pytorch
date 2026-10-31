// Checks the AOTI runtime's CudaGraphManager, which generated code compiles
// into the model .so, against the C shim's CUDA graph functions.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_runtime/cudagraph_manager.h>

#include <algorithm>
#include <array>
#include <memory>
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
    if (sync_stream) {
      // Synchronizing a capturing stream invalidates the capture.
      (void)cudaStreamSynchronize(stream);
      (void)cudaGetLastError();
    }
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
  bool sync_stream = false;
};

at::Tensor affine(const at::Tensor& x) {
  return x * 2 + 1;
}

template <typename Fn>
at::Tensor run(CudaGraphManager& manager, Fn& region, at::Tensor x) {
  auto input = reinterpret_cast<AtenTensorHandle>(&x);
  AtenTensorHandle output = nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  try {
    manager.run(region, &input, 1, &output, 1, stream);
  } catch (...) {
    // A failed call leaves no outputs behind.
    EXPECT_EQ(output, nullptr);
    throw;
  }
  at::Tensor result = tensor_of(output);
  aoti_torch_delete_tensor_object(output);
  return result;
}

CudaGraphOptions no_warmup() {
  CudaGraphOptions options;
  options.warmup_calls = 0;
  return options;
}

template <typename Fn>
void expect_error(Fn&& fn, const std::string& message) {
  try {
    fn();
    ADD_FAILURE() << "expected an error containing: " << message;
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find(message), std::string::npos)
        << e.what();
  }
}

struct CollectWarnings : c10::WarningHandler {
  std::vector<std::string> messages;
  void process(const c10::Warning& warning) override {
    messages.push_back(warning.msg());
  }
};

bool uses_cuda_malloc_async() {
  return c10::cuda::CUDACachingAllocator::name() == "cudaMallocAsync";
}

class CudaGraphManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::cuda::device_count() == 0) {
      GTEST_SKIP() << "CUDA is not available";
    }
    if (uses_cuda_malloc_async()) {
      GTEST_SKIP() << "Capture isn't supported with cudaMallocAsync";
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

TEST_F(CudaGraphManagerTest, NewThreadsWarmUpEvenWithAReusedId) {
  CudaGraphManager manager(0);
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  std::thread([&] { run(manager, region, x); }).join();
  // The new thread often gets the exited one's std::thread::id.
  std::thread([&] {
    run(manager, region, x);
    EXPECT_EQ(manager.num_graphs(), 0);
  }).join();
  EXPECT_EQ(region.calls, 2);
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
  CudaGraphManager manager(0);
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
  // A warm-up call, a capture, then a replay.
  for (int i = 0; i < 3; ++i) {
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
  EXPECT_EQ(calls, 2);
}

TEST_F(CudaGraphManagerTest, RandomRegionsRunEagerly) {
  CollectWarnings warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
  CudaGraphManager manager(0, no_warmup());
  Region region([](const at::Tensor& x) { return at::rand_like(x); });
  auto x = at::empty({1024}, at::kCUDA);
  auto first = run(manager, region, x);
  auto second = run(manager, region, x);
  // A failed capture, then two eager calls.
  EXPECT_EQ(region.calls, 3);
  EXPECT_EQ(manager.num_graphs(), 0);
  EXPECT_FALSE(first.equal(second));
  auto failure = std::find_if(
      warnings.messages.begin(), warnings.messages.end(), [](const auto& msg) {
        return msg.find("capturing failed") != std::string::npos;
      });
  ASSERT_NE(failure, warnings.messages.end());
  EXPECT_NE(failure->find("RNG"), std::string::npos) << *failure;
}

TEST_F(CudaGraphManagerTest, CopiesInputsOfDtypesForeachCopyCantFuse) {
  CudaGraphManager manager(0, no_warmup());
  Region region(
      [](const at::Tensor& x) { return x.view(at::kByte).to(at::kFloat); });
  auto bytes =
      at::randint(0, 255, {8}, at::TensorOptions(at::kCUDA).dtype(at::kByte));
  auto x = bytes.view(at::kFloat8_e8m0fnu);
  run(manager, region, x);
  EXPECT_EQ(manager.num_graphs(), 1);
  bytes.random_(0, 255);
  EXPECT_TRUE(run(manager, region, x).equal(bytes.to(at::kFloat)));

  // An expanded one is copied as the block it covers.
  auto row =
      at::randint(0, 255, {6}, at::TensorOptions(at::kCUDA).dtype(at::kByte));
  auto expanded = row.view(at::kFloat8_e8m0fnu).expand({4, 6});
  run(manager, region, expanded);
  EXPECT_EQ(manager.num_graphs(), 2);
  row.random_(0, 255);
  EXPECT_TRUE(
      run(manager, region, expanded).equal(row.expand({4, 6}).to(at::kFloat)));
}

TEST_F(CudaGraphManagerTest, RejectsACallerCapture) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  at::cuda::CUDAGraph graph;
  {
    at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromPool());
    graph.capture_begin();
    expect_error(
        [&] { run(manager, region, x); }, "inside a CUDA graph capture");
    graph.capture_end();
  }
  EXPECT_EQ(region.calls, 0);
  EXPECT_EQ(manager.num_graphs(), 0);
}

// Run with PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync.
TEST(CudaGraphManagerAsyncTest, RunsEagerlyUnderCudaMallocAsync) {
  if (c10::cuda::device_count() == 0 || !uses_cuda_malloc_async()) {
    GTEST_SKIP() << "Needs the cudaMallocAsync allocator backend";
  }
  CollectWarnings warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  }
  EXPECT_EQ(manager.num_graphs(), 0);
  ASSERT_EQ(warnings.messages.size(), 1);
  EXPECT_NE(
      warnings.messages[0].find("backend:cudaMallocAsync"), std::string::npos);
}

TEST_F(CudaGraphManagerTest, FailedWarmupDoesntCount) {
  CudaGraphManager manager(0);
  Region region(affine);
  region.fail = true;
  auto x = at::randn({16}, at::kCUDA);
  EXPECT_THROW(run(manager, region, x), c10::Error);
  region.fail = false;
  // Still the warm-up call, then the capture.
  run(manager, region, x);
  EXPECT_EQ(manager.num_graphs(), 0);
  run(manager, region, x);
  EXPECT_EQ(manager.num_graphs(), 1);
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

TEST_F(CudaGraphManagerTest, ResetWaitsForTheLastReplay) {
  CudaGraphManager manager(0, no_warmup());
  Region region([](const at::Tensor& x) {
    at::cuda::sleep(1 << 28);
    return affine(x);
  });
  auto x = at::randn({16}, at::kCUDA);
  run(manager, region, x);
  c10::cuda::device_synchronize();
  run(manager, region, x);
  manager.reset();
  EXPECT_EQ(
      cudaStreamQuery(at::cuda::getCurrentCUDAStream().stream()), cudaSuccess);
}

TEST_F(CudaGraphManagerTest, DestructorWaitsForTheLastReplay) {
  Region region([](const at::Tensor& x) {
    at::cuda::sleep(1 << 28);
    return affine(x);
  });
  auto x = at::randn({16}, at::kCUDA);
  {
    CudaGraphManager manager(0, no_warmup());
    run(manager, region, x);
    c10::cuda::device_synchronize();
    run(manager, region, x);
  }
  EXPECT_EQ(
      cudaStreamQuery(at::cuda::getCurrentCUDAStream().stream()), cudaSuccess);
}

TEST_F(CudaGraphManagerTest, RunsConcurrently) {
  // Threads on their own streams share one manager, then each use their own.
  constexpr int kThreads = 4;
  auto region = [](AtenTensorHandle* inputs,
                   AtenTensorHandle* outputs,
                   cudaStream_t stream) {
    const at::Tensor& x = tensor_of(inputs[0]);
    at::cuda::CUDAStreamGuard guard(
        at::cuda::getStreamFromExternal(stream, x.get_device()));
    outputs[0] = new_handle(affine(x));
  };
  CudaGraphManager shared(0);
  std::vector<std::unique_ptr<CudaGraphManager>> own;
  for (int t = 0; t < kThreads; ++t) {
    own.push_back(std::make_unique<CudaGraphManager>(0));
  }
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromPool());
      for (int i = 0; i < 40; ++i) {
        auto& manager = i < 20 ? shared : *own[t];
        auto x = at::randn({8 << (i % 2)}, at::kCUDA);
        EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(shared.num_graphs(), 2);
  for (const auto& manager : own) {
    EXPECT_EQ(manager->num_graphs(), 2);
  }
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

TEST_F(CudaGraphManagerTest, InvalidatedCaptureRunsEagerly) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  region.sync_stream = true;
  auto x = at::randn({16}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(manager.num_graphs(), 0);

  // Other keys still capture.
  region.sync_stream = false;
  auto y = at::randn({8}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, y).equal(affine(y)));
  EXPECT_EQ(manager.num_graphs(), 1);
}

TEST_F(CudaGraphManagerTest, StopsCapturingAfterRepeatedFailures) {
  CollectWarnings warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  region.fail_in_capture = true;
  constexpr int64_t kFailures = CudaGraphManager::kMaxCaptureFailures;
  for (int64_t n = 1; n <= kFailures; ++n) {
    run(manager, region, at::randn({n}, at::kCUDA));
  }
  EXPECT_EQ(region.calls, 2 * kFailures);
  // Once on the first failure, once on giving up.
  ASSERT_EQ(warnings.messages.size(), 2);
  EXPECT_NE(
      warnings.messages[0].find("region can't be captured"), std::string::npos);
  EXPECT_NE(warnings.messages[1].find("in a row"), std::string::npos);
  // New keys now run eagerly without trying to capture.
  region.fail_in_capture = false;
  auto x = at::randn({kFailures + 1}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(region.calls, 2 * kFailures + 1);
  EXPECT_EQ(manager.num_graphs(), 0);
  // reset() starts over.
  manager.reset();
  EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
  EXPECT_EQ(manager.num_graphs(), 1);
}

TEST_F(CudaGraphManagerTest, CaptureFailuresCountOnlyInARow) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  constexpr int64_t kFailures = CudaGraphManager::kMaxCaptureFailures;
  int64_t n = 1;
  auto fail_captures = [&](int64_t count) {
    region.fail_in_capture = true;
    for (int64_t i = 0; i < count; ++i, ++n) {
      run(manager, region, at::randn({n}, at::kCUDA));
    }
    region.fail_in_capture = false;
  };
  fail_captures(kFailures - 1);
  // A successful capture restarts the count.
  run(manager, region, at::randn({n++}, at::kCUDA));
  EXPECT_EQ(manager.num_graphs(), 1);
  fail_captures(kFailures - 1);
  run(manager, region, at::randn({n++}, at::kCUDA));
  EXPECT_EQ(manager.num_graphs(), 2);
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
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  expect_error(
      [&] { manager.run(region, &input, 1, outputs.data(), 2, stream); },
      "different number of outputs");
}

TEST_F(CudaGraphManagerTest, WarnsOnceWhenCaptureFails) {
  CollectWarnings warnings;
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
  const std::string message = "is not on the model's CUDA device";
  expect_error([&] { run(manager, region, at::randn({8})); }, message);
  if (c10::cuda::device_count() >= 2) {
    auto x = at::randn({8}, at::Device(at::kCUDA, 1));
    expect_error([&] { run(manager, region, x); }, message);
  }
  EXPECT_EQ(region.calls, 0);
}

TEST_F(CudaGraphManagerTest, RunsOnItsDeviceWhileAnotherIsCurrent) {
  if (c10::cuda::device_count() < 2) {
    GTEST_SKIP() << "Needs 2 GPUs";
  }
  CudaGraphManager manager(1, no_warmup());
  Region region(affine);
  c10::cuda::CUDAGuard device_guard(0);
  for (int i = 0; i < 3; ++i) {
    auto x = at::randn({8}, at::Device(at::kCUDA, 1));
    auto input = reinterpret_cast<AtenTensorHandle>(&x);
    AtenTensorHandle output = nullptr;
    // The null stream: the legacy default stream of the manager's device.
    manager.run(region, &input, 1, &output, 1, nullptr);
    EXPECT_TRUE(tensor_of(output).equal(affine(x)));
    aoti_torch_delete_tensor_object(output);
  }
  EXPECT_EQ(manager.num_graphs(), 1);
  EXPECT_EQ(region.calls, 1);
  EXPECT_EQ(c10::cuda::current_device(), 0);
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

TEST_F(CudaGraphManagerTest, CopiesInputsOfMixedLayoutsAndDtypes) {
  CudaGraphManager manager(0, no_warmup());
  auto make_inputs = [] {
    return std::vector<at::Tensor>{
        at::randn({8}, at::kCUDA),
        at::randn({6, 4}, at::kCUDA).t(),
        at::randn({1, 6}, at::kCUDA).expand({4, 6}),
        at::randn({6, 8}, at::kCUDA).narrow(1, 2, 3),
        at::randint(100, {5}, at::kCUDA),
        at::randn({7}, at::kCUDA).to(at::kHalf),
        at::randn({}, at::kCUDA)};
  };
  auto expected = [](const std::vector<at::Tensor>& xs) {
    std::vector<at::Tensor> flat;
    for (const auto& x : xs) {
      flat.push_back(x.flatten().to(at::kFloat));
    }
    return at::cat(flat);
  };
  auto region = [&](AtenTensorHandle* inputs,
                    AtenTensorHandle* outputs,
                    cudaStream_t stream) {
    std::vector<at::Tensor> xs;
    for (size_t i = 0; i < 7; ++i) {
      xs.push_back(tensor_of(inputs[i]));
    }
    at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal(stream, 0));
    outputs[0] = new_handle(expected(xs));
  };
  auto run_region = [&](std::vector<at::Tensor> xs) {
    std::vector<AtenTensorHandle> inputs;
    for (auto& x : xs) {
      inputs.push_back(reinterpret_cast<AtenTensorHandle>(&x));
    }
    AtenTensorHandle output = nullptr;
    manager.run(
        region,
        inputs.data(),
        inputs.size(),
        &output,
        1,
        at::cuda::getCurrentCUDAStream().stream());
    at::Tensor result = tensor_of(output);
    aoti_torch_delete_tensor_object(output);
    return result;
  };
  run_region(make_inputs());
  EXPECT_EQ(manager.num_graphs(), 1);
  for (int i = 0; i < 2; ++i) {
    auto xs = make_inputs();
    EXPECT_TRUE(run_region(xs).equal(expected(xs)));
  }
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
  CollectWarnings warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
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
  // Once, however many calls run eagerly past the limit.
  ASSERT_EQ(warnings.messages.size(), 1);
  EXPECT_NE(warnings.messages[0].find("max_graphs=1"), std::string::npos);
}

TEST_F(CudaGraphManagerTest, RejectsNegativeOptions) {
  CudaGraphOptions negative_warmups;
  negative_warmups.warmup_calls = -1;
  expect_error(
      [&] { CudaGraphManager(0, negative_warmups); },
      "warmup_calls must be non-negative");

  CudaGraphOptions negative_max_graphs;
  negative_max_graphs.max_graphs = -1;
  expect_error(
      [&] { CudaGraphManager(0, negative_max_graphs); },
      "max_graphs must be non-negative");
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
