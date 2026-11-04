// Checks the AOTI runtime's CudaGraphManager, which generated code compiles
// into the model .so, against the C shim's CUDA graph functions.

// First, to check that it needs nothing else.
#include <torch/csrc/inductor/aoti_runtime/cudagraph_manager.h>

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
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
using Body = std::function<void(AtenTensorHandle*, AtenTensorHandle*)>;

// A region running on the given stream and counting its calls. By default it
// computes one output from one input with ATen.
struct Region {
  explicit Region(Compute compute)
      : body([compute](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
          outputs[0] = new_handle(compute(tensor_of(inputs[0])));
        }) {}
  explicit Region(Body body) : body(std::move(body)) {}

  void operator()(
      AtenTensorHandle* inputs,
      AtenTensorHandle* outputs,
      cudaStream_t stream) {
    calls++;
    at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal(
        stream, tensor_of(inputs[0]).get_device()));
    body(inputs, outputs);
    if (sync_stream) {
      // Synchronizing a capturing stream invalidates the capture.
      (void)cudaStreamSynchronize(stream);
      (void)cudaGetLastError();
    }
    // A failing region may already have launched work and written outputs.
    TORCH_CHECK(!fail, "region failed");
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    bool capturing = status != cudaStreamCaptureStatusNone;
    TORCH_CHECK(!fail_in_capture || !capturing, "region can't be captured");
    if (oom_in_capture && capturing) {
      // As generated code fails: the error names only the call.
      int64_t size = int64_t(1) << 50;
      int64_t stride = 1;
      AtenTensorHandle huge = nullptr;
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
          1,
          &size,
          &stride,
          aoti_torch_dtype_float32(),
          aoti_torch_device_type_cuda(),
          0,
          &huge));
    }
  }

  Body body;
  std::atomic<int> calls{0};
  bool fail = false;
  bool fail_in_capture = false;
  bool oom_in_capture = false;
  bool sync_stream = false;
};

at::Tensor affine(const at::Tensor& x) {
  return x * 2 + 1;
}

at::Tensor slow_affine(const at::Tensor& x) {
  at::cuda::sleep(1 << 28);
  return affine(x);
}

// The outputs of one run() call, freed with it.
struct Outputs {
  std::vector<AtenTensorHandle> handles;
  ~Outputs() {
    for (auto handle : handles) {
      if (handle != nullptr) {
        aoti_torch_delete_tensor_object(handle);
      }
    }
  }
  const at::Tensor& operator[](size_t i) const {
    return tensor_of(handles[i]);
  }
};

std::unique_ptr<Outputs> run_with(
    CudaGraphManager& manager,
    Region& region,
    std::vector<at::Tensor> inputs,
    size_t num_outputs,
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream()) {
  std::vector<AtenTensorHandle> input_handles;
  for (auto& x : inputs) {
    input_handles.push_back(reinterpret_cast<AtenTensorHandle>(&x));
  }
  auto outputs = std::make_unique<Outputs>();
  // Not null, so a call that fails without nulling a slot is caught.
  auto sentinel = reinterpret_cast<AtenTensorHandle>(outputs.get());
  outputs->handles.resize(num_outputs, sentinel);
  try {
    manager.run(
        region,
        input_handles.data(),
        input_handles.size(),
        outputs->handles.data(),
        num_outputs,
        stream);
  } catch (...) {
    // A failed call leaves no outputs behind.
    for (auto& handle : outputs->handles) {
      EXPECT_EQ(handle, nullptr);
      handle = nullptr;
    }
    throw;
  }
  return outputs;
}

at::Tensor run(CudaGraphManager& manager, Region& region, at::Tensor x) {
  return (*run_with(manager, region, {std::move(x)}, 1))[0];
}

CudaGraphOptions no_warmup(int64_t max_graphs = 8) {
  CudaGraphOptions options;
  options.warmup_calls = 0;
  options.max_graphs = max_graphs;
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

std::atomic<int> foreach_copies{0};
std::atomic<int> copies{0};

// Counts this thread's _foreach_copy_ and copy_ calls while alive. The fused
// kernel calls no copy_; its fallback and per-tensor copies do.
struct CountForeachCopies {
  CountForeachCopies() {
    foreach_copies = copies = 0;
    handle = at::addThreadLocalCallback(
        at::RecordFunctionCallback(
            [](const at::RecordFunction& fn)
                -> std::unique_ptr<at::ObserverContext> {
              std::string_view name = fn.name();
              foreach_copies += name == "aten::_foreach_copy_";
              copies += name == "aten::copy_";
              return nullptr;
            })
            .scopes({at::RecordScope::FUNCTION}));
  }
  ~CountForeachCopies() {
    at::removeCallback(handle);
  }
  at::CallbackHandle handle;
};

struct CollectWarnings : c10::WarningHandler {
  std::vector<std::string> messages;
  void process(const c10::Warning& warning) override {
    messages.push_back(warning.msg());
  }
  bool contains(size_t i, const std::string& text) const {
    return messages.at(i).find(text) != std::string::npos;
  }
};

// Leaves `extra` bytes on top of what the caching allocator holds, until the
// result is destroyed.
auto limit_memory(size_t extra) {
  namespace allocator = c10::cuda::CUDACachingAllocator;
  allocator::emptyCache();
  auto total = at::cuda::getDeviceProperties(0)->totalGlobalMem;
  auto aggregate =
      static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  auto reserved =
      allocator::getDeviceStats(0).reserved_bytes[aggregate].current;
  auto previous = allocator::getMemoryFraction(0);
  allocator::setMemoryFraction(
      static_cast<double>(reserved + extra) / total, 0);
  return c10::make_scope_exit(
      [previous] { allocator::setMemoryFraction(previous, 0); });
}

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
  // A strided view with a storage offset, so the output is not contiguous.
  Region region([](const at::Tensor& x) { return x.narrow(0, 1, 2); });
  auto expected = [](const at::Tensor& x) { return x.narrow(0, 1, 2); };
  auto make_input = [] { return at::randn({6, 4}, at::kCUDA).t(); };

  // A warm-up call runs the region eagerly, so its output can alias the input.
  auto x = make_input();
  EXPECT_TRUE(run(manager, region, x).is_alias_of(x));
  EXPECT_EQ(region.calls, 1);
  EXPECT_EQ(manager.num_graphs(), 0);

  auto y = make_input();
  auto y_out = run(manager, region, y);
  EXPECT_FALSE(y_out.is_alias_of(y));
  EXPECT_TRUE(y_out.equal(expected(y)));
  EXPECT_EQ(y_out.strides(), expected(y).strides());
  EXPECT_EQ(region.calls, 2);
  EXPECT_EQ(manager.num_graphs(), 1);

  auto z = make_input();
  auto z_out = run(manager, region, z);
  EXPECT_FALSE(z_out.is_alias_of(z));
  EXPECT_TRUE(z_out.equal(expected(z)));
  EXPECT_EQ(region.calls, 2);
  // Outputs are clones, so a replay doesn't overwrite earlier results.
  EXPECT_TRUE(y_out.equal(expected(y)));
}

TEST_F(CudaGraphManagerTest, ThreadsRunEagerlyBeforeCapturing) {
  CudaGraphManager manager(0);
  Region region([](const at::Tensor& x) { return x.mm(x); });
  auto x = at::randn({16, 16}, at::kCUDA);
  run(manager, region, x);
  // A new thread has no cuBLAS handle yet, and can't create one in a capture.
  auto y = at::randn({8, 8}, at::kCUDA);
  std::thread([&] {
    region.fail = true;
    expect_error([&] { run(manager, region, x); }, "region failed");
    region.fail = false;
    // Neither a failed call nor another key's eager call warms x up here.
    EXPECT_TRUE(run(manager, region, y).allclose(y.mm(y)));
    EXPECT_TRUE(run(manager, region, x).allclose(x.mm(x)));
    EXPECT_EQ(manager.num_graphs(), 0);
    EXPECT_TRUE(run(manager, region, x).allclose(x.mm(x)));
    EXPECT_EQ(manager.num_graphs(), 1);
  }).join();
  EXPECT_EQ(region.calls, 5);
}

TEST_F(CudaGraphManagerTest, NewThreadsWarmUpEvenWithAReusedId) {
  CudaGraphManager manager(0);
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  std::thread::id first;
  std::thread([&] {
    first = std::this_thread::get_id();
    run(manager, region, x);
  }).join();
  // The new thread often gets the exited one's std::thread::id.
  bool reused = false;
  std::thread([&] {
    reused = std::this_thread::get_id() == first;
    run(manager, region, x);
    EXPECT_EQ(manager.num_graphs(), 0);
  }).join();
  EXPECT_EQ(region.calls, 2);
  if (!reused) {
    GTEST_SKIP() << "the new thread got a new std::thread::id";
  }
}

TEST_F(CudaGraphManagerTest, ClonesOutputsWithTheirLayouts) {
  // Dense outputs of three dtypes, some transposed, one with gaps and one
  // overlapping.
  CudaGraphManager manager(0, no_warmup());
  auto compute = [](const at::Tensor& x) {
    auto e8m0 = (x.abs() * 8).to(at::kByte).view(at::kFloat8_e8m0fnu);
    return std::vector<at::Tensor>{
        x * 2,
        (x + 1).t(),
        x.to(at::kHalf),
        (x * 3).to(at::kHalf).t(),
        e8m0,
        e8m0.t(),
        (x - 1).narrow(1, 1, 2),
        (x * 4).narrow(0, 1, 1).expand({4, 6})};
  };
  Region region([&](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
    auto results = compute(tensor_of(inputs[0]));
    for (size_t i = 0; i < results.size(); ++i) {
      outputs[i] = new_handle(results[i]);
    }
  });
  auto bits = [](const at::Tensor& t) {
    return t.scalar_type() == at::kFloat8_e8m0fnu ? t.view(at::kByte) : t;
  };
  run_with(manager, region, {at::randn({4, 6}, at::kCUDA)}, 8);
  std::vector<std::unique_ptr<Outputs>> outs;
  std::vector<std::vector<at::Tensor>> expected;
  for (int i = 0; i < 2; ++i) {
    auto x = at::randn({4, 6}, at::kCUDA);
    CountForeachCopies counter;
    outs.push_back(run_with(manager, region, {x}, 8));
    // One per dtype of the dense outputs. copy_ runs for the lone input, the
    // two float8_e8m0fnu outputs (the kernel's fallback) and two clones.
    EXPECT_EQ(foreach_copies, 3);
    EXPECT_EQ(copies, 5);
    expected.push_back(compute(x));
  }
  // The first replay's outputs survive the second.
  for (size_t i = 0; i < outs.size(); ++i) {
    for (size_t j = 0; j < expected[i].size(); ++j) {
      EXPECT_TRUE(bits((*outs[i])[j]).equal(bits(expected[i][j]))) << j;
      EXPECT_EQ((*outs[i])[j].strides(), expected[i][j].strides()) << j;
    }
  }
  EXPECT_EQ(region.calls, 1);
}

TEST_F(CudaGraphManagerTest, ReturnsDuplicateAndNullOutputs) {
  CudaGraphManager manager(0);
  Region region([](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
    outputs[0] = new_handle(affine(tensor_of(inputs[0])));
    outputs[1] = outputs[0];
    outputs[2] = nullptr;
  });
  auto x = at::randn({16}, at::kCUDA);
  // A warm-up call, a capture, then a replay.
  for (int i = 0; i < 3; ++i) {
    auto out = run_with(manager, region, {x}, 3);
    EXPECT_NE(out->handles[0], out->handles[1]);
    EXPECT_EQ(out->handles[2], nullptr);
    EXPECT_TRUE((*out)[0].equal(affine(x)));
    EXPECT_TRUE((*out)[1].is_same((*out)[0]));
  }
  EXPECT_EQ(region.calls, 2);
  // Failing calls, warming up or capturing, delete the duplicated handle once.
  auto saved = at::randn({16}, at::kCUDA);
  Region failing_region([&](AtenTensorHandle*, AtenTensorHandle* outputs) {
    outputs[0] = new_handle(saved);
    outputs[1] = outputs[0];
  });
  failing_region.fail = true;
  for (int64_t warmups : {1, 0}) {
    CudaGraphOptions options;
    options.warmup_calls = warmups;
    CudaGraphManager failing(0, options);
    expect_error(
        [&] { run_with(failing, failing_region, {x}, 3); }, "region failed");
    EXPECT_EQ(saved.use_count(), 1) << warmups;
  }
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
  // ATen's warning that the abandoned graph is empty, then the manager's.
  ASSERT_EQ(warnings.messages.size(), 2);
  EXPECT_TRUE(warnings.contains(0, "CUDA Graph is empty"));
  EXPECT_TRUE(warnings.contains(1, "capturing failed"));
  EXPECT_TRUE(warnings.contains(1, "RNG"));
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
  EXPECT_TRUE(warnings.contains(0, "backend:cudaMallocAsync"));
}

TEST_F(CudaGraphManagerTest, FailedCallsAreRethrownAndDontCount) {
  // Not a capture error, so the key warms up and is captured once fixed.
  for (int64_t warmups : {2, 1}) {
    CudaGraphOptions options;
    options.warmup_calls = warmups;
    CudaGraphManager manager(0, options);
    Region region(affine);
    region.fail = true;
    auto x = at::randn({16}, at::kCUDA);
    expect_error([&] { run(manager, region, x); }, "region failed");
    region.fail = false;
    for (int64_t i = 0; i < warmups; ++i) {
      run(manager, region, x);
      EXPECT_EQ(manager.num_graphs(), 0);
    }
    EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
    EXPECT_EQ(manager.num_graphs(), 1);
  }
}

TEST_F(CudaGraphManagerTest, OrdersReplaysOnDifferentStreams) {
  CudaGraphManager manager(0, no_warmup());
  // Long enough that, unordered, the second call's input copy would land
  // while the first replay still reads the shared input buffer.
  Region region(slow_affine);
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
  c10::cuda::device_synchronize();
  auto a_out = run_on(s1, a);
  auto b_out = run_on(s2, b);
  c10::cuda::device_synchronize();
  EXPECT_EQ(region.calls, 1);
  EXPECT_TRUE(a_out.equal(affine(a)));
  EXPECT_TRUE(b_out.equal(affine(b)));
}

TEST_F(CudaGraphManagerTest, RunsConcurrently) {
  // Threads on their own streams share one manager, then each use their own.
  constexpr int kThreads = 4;
  Region region(affine);
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

TEST_F(CudaGraphManagerTest, FailedCaptureRunsEagerlyAndWarnsOnce) {
  // The region throws in the capture, invalidates it, or runs out of memory.
  const char* causes[] = {
      "region can't be captured", "CUDA error", "out of memory"};
  for (int mode = 0; mode < 3; ++mode) {
    CollectWarnings warnings;
    c10::WarningUtils::WarningHandlerGuard guard(&warnings);
    CudaGraphManager manager(0, no_warmup());
    Region region(affine);
    auto z = at::randn({4}, at::kCUDA);
    run(manager, region, z);
    region.fail_in_capture = mode == 0;
    region.sync_stream = mode == 1;
    region.oom_in_capture = mode == 2;
    auto x = at::randn({16}, at::kCUDA);
    EXPECT_TRUE(run(manager, region, x).equal(affine(x)));
    EXPECT_EQ(region.calls, 3);
    ASSERT_EQ(warnings.messages.size(), 1);
    EXPECT_TRUE(warnings.contains(0, causes[mode]));

    // Nothing is captured again until reset(); existing graphs still replay.
    region.fail_in_capture = region.sync_stream = region.oom_in_capture = false;
    auto y = at::randn({8}, at::kCUDA);
    for (const auto& t : {x, y, x, y}) {
      EXPECT_TRUE(run(manager, region, t).equal(affine(t)));
    }
    EXPECT_TRUE(run(manager, region, z).equal(affine(z)));
    EXPECT_EQ(region.calls, 7);
    EXPECT_EQ(manager.num_graphs(), 1);
    EXPECT_EQ(warnings.messages.size(), 1);
    manager.reset();
    run(manager, region, y);
    EXPECT_EQ(manager.num_graphs(), 1);
  }
}

TEST_F(CudaGraphManagerTest, HostOutputsRunEagerly) {
  CollectWarnings warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
  CudaGraphManager manager(0, no_warmup());
  // A replay wouldn't wait for its device-to-host copy.
  Region region([](const at::Tensor& x) {
    auto host = x.options().device(at::kCPU).pinned_memory(true);
    return at::empty(x.sizes(), host).copy_(x, /*non_blocking=*/true);
  });
  auto x = at::randn({16}, at::kCUDA);
  for (int i = 0; i < 2; ++i) {
    auto out = run(manager, region, x);
    EXPECT_TRUE(out.equal(x.cpu()));
  }
  EXPECT_EQ(region.calls, 3);
  EXPECT_EQ(manager.num_graphs(), 0);
  ASSERT_EQ(warnings.messages.size(), 1);
  EXPECT_TRUE(warnings.contains(0, "not on the manager's device"));
}

TEST_F(CudaGraphManagerTest, FailedReplayLeavesNoOutputs) {
  CudaGraphManager manager(0, no_warmup());
  Region region(slow_affine);
  auto x = at::randn({64 << 20}, at::kCUDA);
  run(manager, region, x);
  auto failing_stream = at::cuda::getStreamFromPool();
  {
    // Too little memory for the replay's output clone, while the replay
    // waits behind a long sleep.
    at::cuda::CUDAStreamGuard guard(failing_stream);
    auto restore = limit_memory(32 << 20);
    at::cuda::sleep(1 << 30);
    expect_error([&] { run(manager, region, x); }, "out of memory");
  }
  // The next replay, on another stream, waits for the failed one.
  at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromPool());
  auto y = at::randn_like(x);
  auto out = run(manager, region, y);
  at::cuda::getCurrentCUDAStream().synchronize();
  EXPECT_EQ(cudaStreamQuery(failing_stream), cudaSuccess);
  EXPECT_TRUE(out.equal(affine(y)));
  EXPECT_EQ(region.calls, 1);
}

TEST_F(CudaGraphManagerTest, EagerCallsDontHoldTheLock) {
  CudaGraphManager manager(0, no_warmup(/*max_graphs=*/0));
  std::atomic<int> inside{0};
  Region region([&](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
    // Waits, up to a timeout, until both threads are inside the region.
    ++inside;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (inside < 2 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    EXPECT_EQ(inside, 2);
    outputs[0] = new_handle(affine(tensor_of(inputs[0])));
  });
  auto x = at::randn({16}, at::kCUDA);
  std::thread other([&] { run(manager, region, x); });
  run(manager, region, x);
  other.join();
  EXPECT_EQ(manager.num_graphs(), 0);
}

TEST_F(CudaGraphManagerTest, RejectsADifferentOutputCount) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto x = at::randn({16}, at::kCUDA);
  run(manager, region, x);
  run(manager, region, x);
  ASSERT_EQ(manager.num_graphs(), 1);
  expect_error(
      [&] { run_with(manager, region, {x}, 2); },
      "outputs, but the graph was captured with 1");
}

TEST_F(CudaGraphManagerTest, CapturesAnFnThatLeavesAnotherStreamCurrent) {
  CudaGraphManager manager(0, no_warmup());
  auto original = at::cuda::getCurrentCUDAStream();
  auto side = at::cuda::getStreamFromPool();
  auto fn = [&](AtenTensorHandle* inputs,
                AtenTensorHandle* outputs,
                cudaStream_t stream) {
    {
      at::cuda::CUDAStreamGuard guard(
          at::cuda::getStreamFromExternal(stream, 0));
      outputs[0] = new_handle(affine(tensor_of(inputs[0])));
    }
    // Without a guard, so the capture must still end on its own stream.
    at::cuda::setCurrentCUDAStream(side);
  };
  auto x = at::randn({16}, at::kCUDA);
  auto call = [&] {
    AtenTensorHandle input = new_handle(x);
    AtenTensorHandle output = nullptr;
    manager.run(fn, &input, 1, &output, 1, nullptr);
    aoti_torch_delete_tensor_object(input);
    auto result = tensor_of(output).clone();
    aoti_torch_delete_tensor_object(output);
    return result;
  };
  EXPECT_TRUE(call().equal(affine(x)));
  at::cuda::setCurrentCUDAStream(original);
  EXPECT_EQ(manager.num_graphs(), 1);
  x.normal_();
  EXPECT_TRUE(call().equal(affine(x)));
}

TEST_F(CudaGraphManagerTest, RegionErrorDuringCaptureKeepsOtherGraphs) {
  CudaGraphManager manager(0, no_warmup());
  Region region(affine);
  auto a = at::randn({8}, at::kCUDA);
  auto b = at::randn({4}, at::kCUDA);
  run(manager, region, a);
  // Fails in the capture and again eagerly: rethrown, and capturing stops.
  region.fail = true;
  expect_error([&] { run(manager, region, b); }, "region failed");
  region.fail = false;
  a = at::randn({8}, at::kCUDA);
  b = at::randn({4}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, a).equal(affine(a)));
  EXPECT_TRUE(run(manager, region, b).equal(affine(b)));
  EXPECT_EQ(manager.num_graphs(), 1);
  EXPECT_EQ(region.calls, 4);
}

TEST_F(CudaGraphManagerTest, StopsCapturingBeforeTheEagerRerun) {
  // While one thread reruns a failed capture eagerly, another thread's new key
  // runs eagerly too, and a reset() meanwhile restores capturing.
  CudaGraphManager manager(0, no_warmup());
  std::atomic<bool> rerunning{false};
  std::atomic<bool> release{false};
  Region region([&](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
    auto x = tensor_of(inputs[0]);
    if (x.numel() == 16) {
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      C10_CUDA_CHECK(cudaStreamIsCapturing(
          at::cuda::getCurrentCUDAStream().stream(), &status));
      TORCH_CHECK(status == cudaStreamCaptureStatusNone, "can't be captured");
      rerunning = true;
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (!release && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
      // Not the deadline, which a rerun holding the lock would wait for.
      EXPECT_TRUE(release);
    }
    outputs[0] = new_handle(affine(x));
  });
  auto x = at::randn({16}, at::kCUDA);
  std::thread rerun([&] { run(manager, region, x); });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!rerunning && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  if (!rerunning) {
    release = true;
    rerun.join();
    FAIL() << "the failed capture didn't rerun eagerly";
  }
  auto y = at::randn({8}, at::kCUDA);
  EXPECT_TRUE(run(manager, region, y).equal(affine(y)));
  EXPECT_EQ(manager.num_graphs(), 0);
  manager.reset();
  release = true;
  rerun.join();
  run(manager, region, y);
  EXPECT_EQ(manager.num_graphs(), 1);
  EXPECT_EQ(region.calls, 4);
}

TEST_F(CudaGraphManagerTest, ResetAndDestructorWaitForTheLastReplay) {
  // The region allocates nothing from the pool, so dropping it doesn't sync.
  Region region([](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
    at::cuda::sleep(1 << 28);
    outputs[0] = new_handle(tensor_of(inputs[0]).view({-1}));
  });
  for (bool destroy : {false, true}) {
    auto manager = std::make_unique<CudaGraphManager>(0, no_warmup());
    auto x = at::randn({1 << 20}, at::kCUDA);
    run(*manager, region, x);
    auto replay_stream = at::cuda::getStreamFromPool();
    {
      at::cuda::CUDAStreamGuard guard(replay_stream);
      run(*manager, region, x);
    }
    if (destroy) {
      manager.reset();
    } else {
      manager->reset();
    }
    // Freeing the input buffer earlier would let new work overwrite it.
    EXPECT_EQ(cudaStreamQuery(replay_stream), cudaSuccess) << destroy;
  }
}

TEST_F(CudaGraphManagerTest, RunsOnTheGivenStream) {
  // The current stream stays the default one; each call's input is written
  // on `side` after a sleep, so only work ordered on `side` sees it.
  CudaGraphManager manager(0);
  Region region(affine);
  auto side = at::cuda::getStreamFromPool();
  for (int i = 0; i < 3; ++i) {
    auto x = at::empty({1 << 16}, at::kCUDA);
    {
      at::cuda::CUDAStreamGuard guard(side);
      at::cuda::sleep(1 << 26);
      x.fill_(i + 1);
    }
    auto out = run_with(manager, region, {x}, 1, side.stream());
    side.synchronize();
    EXPECT_TRUE((*out)[0].equal(affine(x))) << i;
  }
  EXPECT_EQ(manager.num_graphs(), 1);
  EXPECT_EQ(region.calls, 2);
}

TEST_F(CudaGraphManagerTest, RejectsInputsOnOtherDevices) {
  CudaGraphManager manager(0);
  Region region(affine);
  // Even during warm-up, which would otherwise run the region eagerly.
  const std::string message = "is not on the manager's device";
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
    // The null stream: the legacy default stream of the manager's device.
    EXPECT_TRUE(
        (*run_with(manager, region, {x}, 1, nullptr))[0].equal(affine(x)));
  }
  EXPECT_EQ(manager.num_graphs(), 1);
  EXPECT_EQ(region.calls, 1);
  EXPECT_EQ(c10::cuda::current_device(), 0);
}

TEST_F(CudaGraphManagerTest, KeysOnAllInputs) {
  CudaGraphManager manager(0, no_warmup());
  Region region([](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
    auto sum = tensor_of(inputs[1]).sum();
    outputs[0] = new_handle(tensor_of(inputs[0]) + sum);
  });
  auto x = at::randn({8}, at::kCUDA);
  run_with(manager, region, {x, at::randn({4}, at::kCUDA)}, 1);
  run_with(manager, region, {x, at::randn({2}, at::kCUDA)}, 1);
  EXPECT_EQ(manager.num_graphs(), 2);

  x = at::randn({8}, at::kCUDA);
  auto y = at::randn({4}, at::kCUDA);
  EXPECT_TRUE((*run_with(manager, region, {x, y}, 1))[0].allclose(x + y.sum()));
  EXPECT_EQ(region.calls, 2);

  // The same sizes with another dtype or other strides get their own graphs.
  CudaGraphManager by_layout(0, no_warmup());
  Region identity(affine);
  auto z = at::randn({4, 4}, at::kCUDA);
  for (const auto& t : {z, z.to(at::kHalf), z.t()}) {
    for (int i = 0; i < 2; ++i) {
      auto out = run(by_layout, identity, t);
      auto expected = affine(t);
      EXPECT_TRUE(out.equal(expected));
      EXPECT_EQ(out.strides(), expected.strides());
    }
  }
  EXPECT_EQ(by_layout.num_graphs(), 3);
  EXPECT_EQ(identity.calls, 3);
}

TEST_F(CudaGraphManagerTest, CopiesInputsOfMixedLayoutsAndDtypes) {
  CudaGraphManager manager(0, no_warmup());
  auto make_inputs = [] {
    // float8_e8m0fnu is one the fused copy kernel doesn't dispatch.
    auto e8m0 = [](at::IntArrayRef sizes) {
      auto bytes = at::TensorOptions(at::kCUDA).dtype(at::kByte);
      return at::randint(0, 255, sizes, bytes).view(at::kFloat8_e8m0fnu);
    };
    return std::vector<at::Tensor>{
        at::randn({8}, at::kCUDA),
        at::randn({6, 4}, at::kCUDA).t(),
        at::randn({1, 6}, at::kCUDA).expand({4, 6}),
        at::randn({3, 6}, at::kCUDA).narrow(0, 1, 1).expand({4, 6}),
        at::randn({6, 8}, at::kCUDA).narrow(1, 2, 3),
        at::randint(100, {5}, at::kCUDA),
        at::randn({7}, at::kCUDA).to(at::kHalf),
        at::randn({}, at::kCUDA),
        at::randn({0, 4}, at::kCUDA),
        e8m0({8}),
        e8m0({1, 6}).expand({4, 6}),
        // Copied alone, being large.
        at::randn({CudaGraphManager::kMaxFusedCopyBytes / 4 + 1}, at::kCUDA),
        // Overlapping and gapped, each copied alone, being alone in its dtype.
        at::randint(100, {1, 6}, at::TensorOptions(at::kCUDA).dtype(at::kInt))
            .expand({4, 6}),
        at::randn({6, 8}, at::kCUDA).to(at::kBFloat16).narrow(1, 2, 3)};
  };
  auto expected = [](const std::vector<at::Tensor>& xs) {
    std::vector<at::Tensor> flat;
    for (const auto& x : xs) {
      bool e8m0 = x.scalar_type() == at::kFloat8_e8m0fnu;
      flat.push_back((e8m0 ? x.view(at::kByte) : x).flatten().to(at::kFloat));
    }
    return at::cat(flat);
  };
  size_t num_inputs = make_inputs().size();
  Region region([&](AtenTensorHandle* inputs, AtenTensorHandle* outputs) {
    std::vector<at::Tensor> xs;
    for (size_t i = 0; i < num_inputs; ++i) {
      xs.push_back(tensor_of(inputs[i]));
    }
    outputs[0] = new_handle(expected(xs));
  });
  run_with(manager, region, make_inputs(), 1);
  EXPECT_EQ(manager.num_graphs(), 1);
  for (int i = 0; i < 2; ++i) {
    auto xs = make_inputs();
    CountForeachCopies counter;
    auto out = run_with(manager, region, xs, 1);
    // The small float and float8_e8m0fnu inputs, of any layout. copy_ runs for
    // the two float8_e8m0fnu inputs (the kernel's fallback), the five inputs
    // copied alone, and the output's clone.
    EXPECT_EQ(foreach_copies, 2);
    EXPECT_EQ(copies, 8);
    EXPECT_TRUE((*out)[0].equal(expected(xs)));
  }
}

TEST_F(CudaGraphManagerTest, MaxGraphsRunsNewKeysEagerly) {
  CollectWarnings warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
  CudaGraphManager manager(0, no_warmup(/*max_graphs=*/1));
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
  // Once, however many calls run eagerly past the limit, until reset().
  ASSERT_EQ(warnings.messages.size(), 1);
  EXPECT_TRUE(warnings.contains(0, "max_graphs=1"));
  manager.reset();
  run(manager, region, a);
  run(manager, region, b);
  EXPECT_EQ(warnings.messages.size(), 2);
  // max_graphs=0 turns capturing off without a warning.
  CudaGraphManager off(0, no_warmup(/*max_graphs=*/0));
  run(off, region, a);
  EXPECT_EQ(warnings.messages.size(), 2);
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

TEST_F(CudaGraphManagerTest, RejectsNegativeOptions) {
  for (bool warmups : {true, false}) {
    CudaGraphOptions options;
    (warmups ? options.warmup_calls : options.max_graphs) = -1;
    expect_error(
        [&] { CudaGraphManager manager(0, options); }, "must be non-negative");
  }
}
