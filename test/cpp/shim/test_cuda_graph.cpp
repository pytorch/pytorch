// Checks the CUDA graph capture functions of the C shim.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

class AOTICudaGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::cuda::device_count() == 0) {
      GTEST_SKIP() << "CUDA is not available";
    }
    ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_graph(&graph_), AOTI_TORCH_SUCCESS);
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
  at::Tensor capture(Fn fn, CUDAGraphHandle graph = nullptr) {
    graph = graph != nullptr ? graph : graph_;
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_begin(
            graph, pool_, capture_stream_->stream()),
        AOTI_TORCH_SUCCESS);
    at::Tensor out;
    {
      at::cuda::CUDAStreamGuard guard(*capture_stream_);
      out = fn();
    }
    EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph), AOTI_TORCH_SUCCESS);
    return out;
  }

  bool stream_is_capturing() {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    EXPECT_EQ(
        cudaStreamIsCapturing(capture_stream_->stream(), &status), cudaSuccess);
    return status != cudaStreamCaptureStatusNone;
  }

  // Checks that the stream and the allocator are left fit for a new capture,
  // into a new pool since graph_ was the pool's only graph.
  void expect_new_capture_works() {
    if (graph_ != nullptr) {
      ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
    }
    ASSERT_EQ(aoti_torch_delete_cuda_mem_pool(pool_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_graph(&graph_), AOTI_TORCH_SUCCESS);
    auto x = at::zeros({8}, at::kCUDA);
    auto y = capture([&] { return x + 1; });
    x.fill_(2);
    ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
    EXPECT_TRUE(y.equal(at::full({8}, 3.0, at::kCUDA)));
  }

  AOTITorchError replay(CUDAGraphHandle graph = nullptr) {
    return aoti_torch_cuda_graph_replay(
        graph != nullptr ? graph : graph_,
        at::cuda::getCurrentCUDAStream().stream());
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
  for (bool delete_first_graph_first : {true, false}) {
    ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_graph(&graph_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_delete_cuda_mem_pool(pool_), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool_), AOTI_TORCH_SUCCESS);
    CUDAGraphHandle second = nullptr;
    ASSERT_EQ(aoti_torch_create_cuda_graph(&second), AOTI_TORCH_SUCCESS);
    auto x = at::zeros({8}, at::kCUDA);
    auto y1 = capture([&] { return x + 1; });
    auto y2 = capture([&] { return x * 2; }, second);
    x.fill_(3);
    ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(replay(second), AOTI_TORCH_SUCCESS);
    EXPECT_TRUE(y1.equal(at::full({8}, 4.0, at::kCUDA)));
    EXPECT_TRUE(y2.equal(at::full({8}, 6.0, at::kCUDA)));

    y1.reset();
    y2.reset();
    auto* last = second;
    if (delete_first_graph_first) {
      ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
    } else {
      ASSERT_EQ(aoti_torch_delete_cuda_graph(second), AOTI_TORCH_SUCCESS);
      last = graph_;
    }
    x.fill_(5);
    ASSERT_EQ(replay(last), AOTI_TORCH_SUCCESS);
    ASSERT_EQ(aoti_torch_delete_cuda_graph(last), AOTI_TORCH_SUCCESS);
    // Every graph captured into pool_ is gone, so it takes no more captures.
    ASSERT_EQ(aoti_torch_create_cuda_graph(&graph_), AOTI_TORCH_SUCCESS);
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_begin(
            graph_, pool_, capture_stream_->stream()),
        AOTI_TORCH_FAILURE);
    EXPECT_FALSE(stream_is_capturing());
    expect_new_capture_works();
  }
}

TEST_F(AOTICudaGraphTest, PoolTakesOneCaptureAtATime) {
  CUDAGraphHandle second = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_graph(&second), AOTI_TORCH_SUCCESS);
  auto other_stream = at::cuda::getStreamFromPool();
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(second, pool_, other_stream.stream()),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_SUCCESS);
  // Once the first capture ended, the second graph can share the pool.
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; }, second);
  x.fill_(1);
  ASSERT_EQ(replay(second), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::full({8}, 2.0, at::kCUDA)));
  EXPECT_EQ(aoti_torch_delete_cuda_graph(second), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, GraphCapturesOnce) {
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; });
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_FAILURE);
  EXPECT_FALSE(stream_is_capturing());
  // The pool still takes captures from other graphs.
  CUDAGraphHandle second = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_graph(&second), AOTI_TORCH_SUCCESS);
  auto z = capture([&] { return x * 2; }, second);
  x.fill_(2);
  ASSERT_EQ(replay(second), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(z.equal(at::full({8}, 4.0, at::kCUDA)));
  EXPECT_EQ(aoti_torch_delete_cuda_graph(second), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, RejectsCapturingStream) {
  CUDAMemPoolHandle outer_pool = nullptr;
  CUDAGraphHandle outer = nullptr;
  ASSERT_EQ(
      aoti_torch_create_cuda_mem_pool(0, &outer_pool), AOTI_TORCH_SUCCESS);
  ASSERT_EQ(aoti_torch_create_cuda_graph(&outer), AOTI_TORCH_SUCCESS);
  auto stream = capture_stream_->stream();
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(outer, outer_pool, stream),
      AOTI_TORCH_SUCCESS);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool_, stream),
      AOTI_TORCH_FAILURE);
  // The outer capture is still valid, and pool_ and graph_ are untouched.
  EXPECT_EQ(aoti_torch_delete_cuda_graph(outer), AOTI_TORCH_SUCCESS);
  EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(outer_pool), AOTI_TORCH_SUCCESS);
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; });
  x.fill_(1);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::full({8}, 2.0, at::kCUDA)));
}

TEST_F(AOTICudaGraphTest, RejectsBlockingStreams) {
  cudaStream_t blocking = nullptr;
  ASSERT_EQ(cudaStreamCreate(&blocking), cudaSuccess);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool_, blocking),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool_, nullptr),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(cudaStreamDestroy(blocking), cudaSuccess);
  // Neither attempt touched the pool or the graph.
  auto x = at::zeros({8}, at::kCUDA);
  auto y = capture([&] { return x + 1; });
  x.fill_(1);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::full({8}, 2.0, at::kCUDA)));
}

TEST_F(AOTICudaGraphTest, OtherThreadsRunDuringCapture) {
  auto x = at::zeros({8}, at::kCUDA);
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  at::Tensor y;
  {
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
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
  ASSERT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_SUCCESS);
  x.fill_(2);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  EXPECT_TRUE(y.equal(at::full({8}, 3.0, at::kCUDA)));
}

TEST_F(AOTICudaGraphTest, NullHandlesFail) {
  auto stream = capture_stream_->stream();
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(nullptr, pool_, stream),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, nullptr, stream),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(aoti_torch_cuda_graph_capture_end(nullptr), AOTI_TORCH_FAILURE);
  EXPECT_EQ(aoti_torch_cuda_graph_replay(nullptr, stream), AOTI_TORCH_FAILURE);
  EXPECT_EQ(aoti_torch_delete_cuda_graph(nullptr), AOTI_TORCH_SUCCESS);
  EXPECT_FALSE(stream_is_capturing());
}

TEST_F(AOTICudaGraphTest, AbandoningEmptyCaptureDoesntWarn) {
  struct CollectWarnings : c10::WarningHandler {
    std::vector<std::string> messages;
    void process(const c10::Warning& warning) override {
      messages.push_back(warning.msg());
    }
  } warnings;
  c10::WarningUtils::WarningHandlerGuard guard(&warnings);
  auto stream = capture_stream_->stream();
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool_, stream),
      AOTI_TORCH_SUCCESS);
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  graph_ = nullptr;
  EXPECT_TRUE(warnings.messages.empty());
  // Ending an empty capture does warn, so the filter still matches.
  CUDAMemPoolHandle pool = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_mem_pool(0, &pool), AOTI_TORCH_SUCCESS);
  ASSERT_EQ(aoti_torch_create_cuda_graph(&graph_), AOTI_TORCH_SUCCESS);
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool, stream),
      AOTI_TORCH_SUCCESS);
  (void)aoti_torch_cuda_graph_capture_end(graph_);
  ASSERT_EQ(warnings.messages.size(), 1);
  EXPECT_NE(
      warnings.messages[0].find("The CUDA Graph is empty"), std::string::npos);
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  graph_ = nullptr;
  EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, UsesThePoolsDevice) {
  if (c10::cuda::device_count() < 2) {
    GTEST_SKIP() << "Needs 2 GPUs";
  }
  CUDAMemPoolHandle pool = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_mem_pool(1, &pool), AOTI_TORCH_SUCCESS);
  auto stream = at::cuda::getStreamFromPool(false, 1);
  c10::cuda::CUDAGuard device_guard(0);
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool, stream.stream()),
      AOTI_TORCH_SUCCESS);
  at::Tensor y;
  {
    at::cuda::CUDAStreamGuard guard(stream);
    y = at::rand({1024}, at::Device(at::kCUDA, 1));
  }
  ASSERT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_SUCCESS);
  EXPECT_EQ(c10::cuda::current_device(), 0);
  auto replay_stream = at::cuda::getStreamFromPool(false, 1);
  ASSERT_EQ(
      aoti_torch_cuda_graph_replay(graph_, replay_stream.stream()),
      AOTI_TORCH_SUCCESS);
  replay_stream.synchronize();
  auto first = y.clone();
  ASSERT_EQ(
      aoti_torch_cuda_graph_replay(graph_, replay_stream.stream()),
      AOTI_TORCH_SUCCESS);
  replay_stream.synchronize();
  EXPECT_FALSE(y.equal(first));
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  graph_ = nullptr;
  EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, CaptureEndsOnlyOnItsThread) {
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  std::thread([&] {
    EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_FAILURE);
    EXPECT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_FAILURE);
  }).join();
  EXPECT_TRUE(stream_is_capturing());
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  graph_ = nullptr;
  EXPECT_FALSE(stream_is_capturing());
  expect_new_capture_works();
}

TEST_F(AOTICudaGraphTest, EndingInvalidatedCaptureFailsButEndsIt) {
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  // Synchronizing a capturing stream is illegal and invalidates the capture.
  EXPECT_NE(cudaStreamSynchronize(capture_stream_->stream()), cudaSuccess);
  (void)cudaGetLastError();
  EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_FAILURE);
  EXPECT_FALSE(stream_is_capturing());
  EXPECT_EQ(replay(), AOTI_TORCH_FAILURE);
  expect_new_capture_works();
}

TEST_F(AOTICudaGraphTest, DeletingInvalidatedCaptureEndsIt) {
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  EXPECT_NE(cudaStreamSynchronize(capture_stream_->stream()), cudaSuccess);
  (void)cudaGetLastError();
  // Reports the capture's error, but deletes the graph.
  EXPECT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_FAILURE);
  graph_ = nullptr;
  EXPECT_FALSE(stream_is_capturing());
  expect_new_capture_works();
}

TEST_F(AOTICudaGraphTest, DeletingDuringCaptureEndsIt) {
  auto x = at::zeros({8}, at::kCUDA);
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  {
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    x.add_(1);
  }
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  graph_ = nullptr;
  EXPECT_FALSE(stream_is_capturing());
  expect_new_capture_works();
}

TEST_F(AOTICudaGraphTest, OutOfOrderCallsFail) {
  EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_FAILURE);
  EXPECT_EQ(replay(), AOTI_TORCH_FAILURE);
}
