// Checks the CUDA graph capture functions of the C shim and their RAII
// wrappers in utils_cuda.h.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <algorithm>
#include <optional>
#include <string>
#include <thread>

namespace {

constexpr int32_t kThreadLocal = cudaStreamCaptureModeThreadLocal;

bool uses_cuda_malloc_async() {
  return c10::cuda::CUDACachingAllocator::name() == "cudaMallocAsync";
}

class AOTICudaGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::cuda::device_count() == 0) {
      GTEST_SKIP() << "CUDA is not available";
    }
    if (uses_cuda_malloc_async()) {
      GTEST_SKIP() << "Capture isn't supported with cudaMallocAsync";
    }
    ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_graph(0, 8, &graph_), AOTI_TORCH_SUCCESS);
    capture_stream_ = at::cuda::getStreamFromPool();
  }

  void TearDown() override {
    if (graph_ != nullptr) {
      EXPECT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
    }
    if (pool_ != nullptr) {
      EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool_), AOTI_TORCH_SUCCESS);
    }
  }

  // Captures fn's work into graph (default graph_) on the capture stream and
  // returns its result.
  template <typename Fn>
  at::Tensor capture(
      Fn fn,
      CUDAGraphHandle graph = nullptr,
      int32_t mode = kThreadLocal,
      CUDAMemPoolHandle pool = nullptr,
      std::optional<at::cuda::CUDAStream> stream = std::nullopt) {
    graph = graph != nullptr ? graph : graph_;
    pool = pool != nullptr ? pool : pool_;
    at::cuda::CUDAStreamGuard guard(stream.value_or(*capture_stream_));
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_begin(graph, pool, mode),
        AOTI_TORCH_SUCCESS);
    at::Tensor out;
    try {
      out = fn();
    } catch (...) {
      // TearDown deletes the pool, which an open capture forbids.
      (void)aoti_torch_cuda_graph_capture_end(graph);
      throw;
    }
    EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph), AOTI_TORCH_SUCCESS);
    return out;
  }

  bool stream_is_capturing() {
    return capture_stream_->is_capturing();
  }

  AOTITorchError replay(CUDAGraphHandle graph = nullptr) {
    return aoti_torch_cuda_graph_replay(graph != nullptr ? graph : graph_);
  }

  CUDAMemPoolHandle pool_ = nullptr;
  CUDAGraphHandle graph_ = nullptr;
  std::optional<at::cuda::CUDAStream> capture_stream_;
};

} // namespace

TEST_F(AOTICudaGraphTest, CapturesAndReplays) {
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x * 2 + 1; });
  x.copy_(at::randn({8}, at::kCUDA));
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(x * 2 + 1));
}

TEST_F(AOTICudaGraphTest, CapturesInEachMode) {
  for (int32_t mode :
       {cudaStreamCaptureModeGlobal,
        cudaStreamCaptureModeThreadLocal,
        cudaStreamCaptureModeRelaxed}) {
    // A deleted graph's pool can't take new captures, so each mode starts
    // over.
    ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_delete_cuda_mem_pool(pool_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_graph(0, 8, &graph_), AOTI_TORCH_SUCCESS);
    auto x = at::zeros({8}, at::kCUDA);
    auto y = capture([&] { return x + 1; }, graph_, mode);
    x.fill_(3);
    ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
    EXPECT_TRUE(y.equal(at::full({8}, 4.0, at::kCUDA)));
  }
}

TEST_F(AOTICudaGraphTest, ReplaysAdvanceRng) {
  auto y = capture([] { return at::rand({1024}, at::kCUDA); });
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  auto first = y.clone();
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_FALSE(y.equal(first));
}

TEST_F(AOTICudaGraphTest, GraphKeepsDeletedPoolsMemory) {
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; });
  ASSERT_EQ(aoti_torch_delete_cuda_mem_pool(pool_), AOTI_TORCH_SUCCESS);
  pool_ = nullptr;
  x.fill_(3);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::full({8}, 4.0, at::kCUDA)));
}

TEST_F(AOTICudaGraphTest, GraphsShareAPool) {
  CUDAGraphHandle second = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_graph(0, 8, &second), AOTI_TORCH_SUCCESS);
  auto x = at::zeros({8}, at::kCUDA);
  auto y1 = capture([&] { return x + 1; });
  auto y2 = capture([&] { return x * 2; }, second);
  x.fill_(3);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y1.equal(at::full({8}, 4.0, at::kCUDA)));
  ASSERT_EQ(replay(second), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y2.equal(at::full({8}, 6.0, at::kCUDA)));
  EXPECT_EQ(aoti_torch_delete_cuda_graph(second), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, KeepsTheGraph) {
  CUDAGraphHandle kept = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_graph(1, 8, &kept), AOTI_TORCH_SUCCESS);
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; }, kept);
  for (double v : {3.0, 5.0}) {
    x.fill_(v);
    ASSERT_EQ(replay(kept), AOTI_TORCH_SUCCESS);
    EXPECT_TRUE(y.equal(at::full({8}, v + 1, at::kCUDA)));
  }
  EXPECT_EQ(aoti_torch_delete_cuda_graph(kept), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, ReplaysOnTheCurrentStream) {
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; });
  auto stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard guard(stream);
  // Holds the stream back, so the replay must wait behind the fill.
  at::cuda::sleep(100 * 1000 * 1000);
  x.fill_(5);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  stream.synchronize();
  EXPECT_TRUE(y.equal(at::full({8}, 6.0, at::kCUDA)));
}

TEST_F(AOTICudaGraphTest, OtherThreadsRunDuringCapture) {
  auto x = at::zeros({8}, at::kCUDA);
  at::Tensor y;
  {
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    ASSERT_EQ(
        aoti_torch_cuda_graph_capture_begin(graph_, pool_, kThreadLocal),
        AOTI_TORCH_SUCCESS);
    y = x + 1;
  }
  // Another thread allocates, launches and syncs on its own stream and on the
  // legacy default stream while the capture is open.
  bool other_thread_ok = false;
  std::thread([&] {
    auto stream = at::cuda::getStreamFromPool();
    at::Tensor a, b, c;
    {
      at::cuda::CUDAStreamGuard guard(stream);
      a = at::randn({1024}, at::kCUDA);
      b = a * 2;
      stream.synchronize();
    }
    {
      at::cuda::CUDAStreamGuard guard(at::cuda::getDefaultCUDAStream());
      c = a + 1;
      other_thread_ok =
          b.cpu().equal(a.cpu() * 2) && c.cpu().equal(a.cpu() + 1);
    }
  }).join();
  EXPECT_TRUE(other_thread_ok);
  EXPECT_TRUE(stream_is_capturing());
  {
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    ASSERT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_SUCCESS);
  }
  x.fill_(2);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::full({8}, 3.0, at::kCUDA)));
}

TEST_F(AOTICudaGraphTest, UsesThePoolsDevice) {
  if (c10::cuda::device_count() < 2) {
    GTEST_SKIP() << "Needs 2 GPUs";
  }
  CUDAMemPoolHandle pool = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_mem_pool(1, &pool), AOTI_TORCH_SUCCESS);
  {
    // A pool on device 1 while device 0 is current.
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_begin(graph_, pool, kThreadLocal),
        AOTI_TORCH_FAILURE);
    EXPECT_FALSE(stream_is_capturing());
  }
  // The capture uses device 1's default generator, which nothing created.
  auto y = capture(
      [] { return at::rand({1024}, at::Device(at::kCUDA, 1)); },
      graph_,
      kThreadLocal,
      pool,
      at::cuda::getStreamFromPool(false, 1));
  {
    // A replay runs on the current stream of the capture's device.
    c10::cuda::CUDAGuard device_guard(0);
    auto replay_stream = at::cuda::getStreamFromPool(false, 1);
    c10::cuda::CUDAMultiStreamGuard stream_guard{
        c10::ArrayRef<c10::cuda::CUDAStream>(replay_stream)};
    ASSERT_EQ(at::cuda::getCurrentCUDAStream(1), replay_stream);
    ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
    replay_stream.synchronize();
    auto first = y.clone();
    ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
    replay_stream.synchronize();
    EXPECT_FALSE(y.equal(first));
  }
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  graph_ = nullptr;
  EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, DeletingAPoolWhileCapturingFails) {
  CUDAMemPoolHandle pool = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool), AOTI_TORCH_SUCCESS);
  {
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    ASSERT_EQ(
        aoti_torch_cuda_graph_capture_begin(graph_, pool_, kThreadLocal),
        AOTI_TORCH_SUCCESS);
    EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool), AOTI_TORCH_FAILURE);
    EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_SUCCESS);
  }
  EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, ChecksTheirArguments) {
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, nullptr, kThreadLocal),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(nullptr, pool_, kThreadLocal),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(aoti_torch_cuda_graph_capture_end(nullptr), AOTI_TORCH_FAILURE);
  EXPECT_EQ(aoti_torch_cuda_graph_replay(nullptr), AOTI_TORCH_FAILURE);
  {
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_begin(graph_, pool_, 3),
        AOTI_TORCH_FAILURE);
    EXPECT_FALSE(stream_is_capturing());
  }
  // The rejected capture didn't touch the graph.
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; });
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::ones({8}, at::kCUDA)));
  CUDAGraphHandle graph = nullptr;
  EXPECT_EQ(aoti_torch_create_cuda_graph(0, -1, &graph), AOTI_TORCH_FAILURE);
  EXPECT_EQ(graph, nullptr);
}

// Run with PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync.
TEST(AOTICudaGraphAsyncTest, RejectsTheCudaMallocAsyncBackend) {
  if (c10::cuda::device_count() == 0 || !uses_cuda_malloc_async()) {
    GTEST_SKIP() << "Needs the cudaMallocAsync allocator backend";
  }
  CUDAMemPoolHandle pool = nullptr;
  EXPECT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool), AOTI_TORCH_FAILURE);
  EXPECT_EQ(pool, nullptr);
}

TEST_F(AOTICudaGraphTest, EndingInvalidatedCaptureFailsButEndsIt) {
  at::cuda::CUDAStreamGuard guard(*capture_stream_);
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool_, kThreadLocal),
      AOTI_TORCH_SUCCESS);
  // Synchronizing a capturing stream is illegal and invalidates the capture.
  EXPECT_NE(cudaStreamSynchronize(capture_stream_->stream()), cudaSuccess);
  (void)cudaGetLastError();
  EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_FAILURE);
  EXPECT_FALSE(stream_is_capturing());
  EXPECT_EQ(replay(), AOTI_TORCH_FAILURE);
  // The stream and a new pool take a new capture.
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  ASSERT_EQ(aoti_torch_delete_cuda_mem_pool(pool_), AOTI_TORCH_SUCCESS);
  ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool_), AOTI_TORCH_SUCCESS);
  ASSERT_EQ(aoti_torch_create_cuda_graph(0, 8, &graph_), AOTI_TORCH_SUCCESS);
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; });
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::ones({8}, at::kCUDA)));
}

TEST_F(AOTICudaGraphTest, ReplayBeforeCaptureFails) {
  EXPECT_EQ(replay(), AOTI_TORCH_FAILURE);
}

namespace {

class AOTICudaGraphRAIITest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::cuda::device_count() == 0 || uses_cuda_malloc_async()) {
      GTEST_SKIP() << "Needs CUDA and the native caching allocator";
    }
    pool_.emplace(0);
    stream_.emplace(0);
  }

  // Deleting the pool with a capture open aborts, so a test whose cleanup
  // regressed fails here rather than taking down the binary.
  void TearDown() override {
    if (stream_) {
      end_leftover_capture(stream_->get());
    }
  }

  static void end_leftover_capture(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
        status != cudaStreamCaptureStatusNone) {
      ADD_FAILURE() << "a capture was left open";
      cudaGraph_t graph = nullptr;
      (void)cudaStreamEndCapture(stream, &graph);
      if (graph != nullptr) {
        (void)cudaGraphDestroy(graph);
      }
    }
    (void)cudaGetLastError();
  }

  template <typename Fn>
  static std::string error_of(Fn&& fn) {
    try {
      fn();
    } catch (const std::exception& e) {
      return e.what();
    }
    return "no error";
  }

  bool stream_is_capturing() {
    return at::cuda::getStreamFromExternal(stream_->get(), 0).is_capturing();
  }

  // Makes the test's stream current.
  at::cuda::CUDAStreamGuard on_stream() {
    return at::cuda::CUDAStreamGuard(
        at::cuda::getStreamFromExternal(stream_->get(), 0));
  }

  std::optional<torch::aot_inductor::AOTICudaMemPool> pool_;
  std::optional<torch::aot_inductor::AOTICudaStreamNonBlocking> stream_;
};

} // namespace

TEST_F(AOTICudaGraphRAIITest, CapturesAndReplays) {
  torch::aot_inductor::AOTICudaGraph graph;
  auto x = at::ones({8}, at::kCUDA);
  at::Tensor y;
  {
    auto guard = on_stream();
    graph.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
    y = x * 2;
    graph.capture_end();
  }
  x.fill_(3);
  graph.replay();
  EXPECT_TRUE(y.equal(at::full({8}, 6.0f, at::kCUDA)));
  auto what = error_of([] { torch::aot_inductor::AOTICudaGraph().replay(); });
  EXPECT_NE(what.find("aoti_torch_cuda_graph_replay("), what.npos) << what;
}

TEST_F(AOTICudaGraphRAIITest, EndsTheCaptureItLeftOpen) {
  {
    auto guard = on_stream();
    torch::aot_inductor::AOTICudaGraph graph;
    graph.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
    auto y = at::ones({8}, at::kCUDA) * 2;
    // Leaves another stream current, as captured code may, when the graph is
    // destroyed.
    at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromPool());
  }
  EXPECT_FALSE(stream_is_capturing());
}

TEST_F(AOTICudaGraphRAIITest, EndsTheCaptureOnItsStream) {
  torch::aot_inductor::AOTICudaGraph graph;
  at::Tensor y;
  {
    auto guard = on_stream();
    graph.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
    y = at::ones({8}, at::kCUDA) * 2;
    // As captured code may leave another stream current.
    at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromPool());
    graph.capture_end();
  }
  EXPECT_FALSE(stream_is_capturing());
  graph.replay();
  EXPECT_TRUE(y.equal(at::full({8}, 2.0f, at::kCUDA)));
}

TEST_F(AOTICudaGraphRAIITest, RejectsANegativeDeviceIndex) {
  EXPECT_EQ(
      error_of([] { torch::aot_inductor::AOTICudaMemPool(-1); }),
      "AOTICudaMemPool: device_index must be >= 0, got -1");
}

TEST_F(AOTICudaGraphRAIITest, EndsOnlyACaptureItBegan) {
  torch::aot_inductor::AOTICudaGraph graph;
  EXPECT_EQ(
      error_of([&] { graph.capture_end(); }),
      "AOTICudaGraph: no capture to end");
  {
    auto guard = on_stream();
    graph.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
    auto y = at::ones({8}, at::kCUDA) * 2;
    graph.capture_end();
  }
  EXPECT_EQ(
      error_of([&] { graph.capture_end(); }),
      "AOTICudaGraph: no capture to end");
}

TEST_F(AOTICudaGraphRAIITest, MakesNonBlockingStreamsAndEvents) {
  for (int32_t device = 0;
       device < std::min<int32_t>(c10::cuda::device_count(), 2);
       ++device) {
    torch::aot_inductor::AOTICudaStreamNonBlocking stream(device);
    unsigned int flags = 0;
    ASSERT_EQ(cudaStreamGetFlags(stream.get(), &flags), cudaSuccess);
    EXPECT_EQ(flags, cudaStreamNonBlocking);
    // With device 0 current, recording fails unless the event and the stream
    // share a device, so check each wrapper against one known to be on it.
    torch::aot_inductor::AOTICudaEventDisableTiming event(device);
    auto known_stream = at::cuda::getStreamFromPool(false, device);
    EXPECT_EQ(cudaEventRecord(event.get(), known_stream.stream()), cudaSuccess);
    at::cuda::CUDAEvent known_event;
    known_event.record(at::cuda::getStreamFromExternal(stream.get(), device));
    EXPECT_EQ(cudaEventSynchronize(event.get()), cudaSuccess);
    known_event.synchronize();
    // The event has no timing data.
    float ms = 0;
    EXPECT_EQ(
        cudaEventElapsedTime(&ms, event.get(), event.get()),
        cudaErrorInvalidResourceHandle);
    (void)cudaGetLastError();
  }
}

TEST_F(AOTICudaGraphRAIITest, LeavesACaptureItFailedToBeginAlone) {
  auto guard = on_stream();
  torch::aot_inductor::AOTICudaGraph first;
  first.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
  auto y = at::ones({8}, at::kCUDA) * 2;
  {
    // The shim rejects the mode before it begins a capture.
    torch::aot_inductor::AOTICudaGraph second;
    auto what = error_of([&] {
      second.capture_begin(*pool_, static_cast<cudaStreamCaptureMode>(3));
    });
    EXPECT_NE(what.find("aoti_torch_cuda_graph_capture_begin("), what.npos)
        << what;
  }
  EXPECT_TRUE(stream_is_capturing());
  first.capture_end();
  first.replay();
  EXPECT_TRUE(y.equal(at::full({8}, 2.0f, at::kCUDA)));
}

TEST_F(AOTICudaGraphRAIITest, LeavesALaterCaptureOnTheStreamAlone) {
  auto guard = on_stream();
  auto first = std::make_unique<torch::aot_inductor::AOTICudaGraph>();
  first->capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
  // Invalidates the capture, so capture_end fails after ending it.
  EXPECT_NE(cudaStreamSynchronize(stream_->get()), cudaSuccess);
  (void)cudaGetLastError();
  auto what = error_of([&] { first->capture_end(); });
  EXPECT_NE(what.find("aoti_torch_cuda_graph_capture_end("), what.npos) << what;
  torch::aot_inductor::AOTICudaGraph second;
  second.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
  auto y = at::ones({8}, at::kCUDA) * 2;
  first.reset();
  second.capture_end();
  second.replay();
  EXPECT_TRUE(y.equal(at::full({8}, 2.0f, at::kCUDA)));
}

TEST_F(AOTICudaGraphRAIITest, EndsTheInvalidatedCaptureItLeftOpen) {
  {
    auto guard = on_stream();
    torch::aot_inductor::AOTICudaGraph graph;
    graph.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
    auto x = at::ones({8}, at::kCUDA) * 2;
    // As when the captured code throws after a call that broke the capture.
    EXPECT_NE(cudaStreamSynchronize(stream_->get()), cudaSuccess);
    (void)cudaGetLastError();
  }
  ASSERT_FALSE(stream_is_capturing());
  // The stream captures again, into a new pool: a pool can't take a capture
  // once every graph captured into it is gone.
  pool_.emplace(0);
  auto guard = on_stream();
  torch::aot_inductor::AOTICudaGraph graph;
  graph.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
  auto y = at::ones({8}, at::kCUDA) * 3;
  graph.capture_end();
  graph.replay();
  EXPECT_TRUE(y.equal(at::full({8}, 3.0f, at::kCUDA)));
}

TEST_F(AOTICudaGraphRAIITest, EndsACaptureOnAnotherDevice) {
  if (c10::cuda::device_count() < 2) {
    GTEST_SKIP() << "Needs two GPUs";
  }
  torch::aot_inductor::AOTICudaMemPool pool(1);
  torch::aot_inductor::AOTICudaStreamNonBlocking stream(1);
  auto graph = std::make_unique<torch::aot_inductor::AOTICudaGraph>();
  {
    c10::cuda::CUDAGuard device_guard(1);
    at::cuda::CUDAStreamGuard stream_guard(
        at::cuda::getStreamFromExternal(stream.get(), 1));
    graph->capture_begin(pool, cudaStreamCaptureModeThreadLocal);
  }
  // Destroyed with device 0 current.
  graph.reset();
  EXPECT_FALSE(at::cuda::getStreamFromExternal(stream.get(), 1).is_capturing());
  end_leftover_capture(stream.get());
}
