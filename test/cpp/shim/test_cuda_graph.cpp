// Checks the CUDA graph capture functions of the C shim.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <optional>
#include <thread>
#include <vector>

namespace {

constexpr int32_t kThreadLocal = cudaStreamCaptureModeThreadLocal;

class AOTICudaGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::cuda::device_count() == 0) {
      GTEST_SKIP() << "CUDA is not available";
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
      int32_t mode = kThreadLocal) {
    graph = graph != nullptr ? graph : graph_;
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_begin(graph, pool_, mode),
        AOTI_TORCH_SUCCESS);
    auto out = fn();
    EXPECT_EQ(aoti_torch_cuda_graph_capture_end(graph), AOTI_TORCH_SUCCESS);
    return out;
  }

  bool stream_is_capturing() {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    EXPECT_EQ(
        cudaStreamIsCapturing(capture_stream_->stream(), &status), cudaSuccess);
    return status != cudaStreamCaptureStatusNone;
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
  at::Tensor y;
  {
    // The capture uses device 1's default generator, which nothing created.
    at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromPool(false, 1));
    ASSERT_EQ(
        aoti_torch_cuda_graph_capture_begin(graph_, pool, kThreadLocal),
        AOTI_TORCH_SUCCESS);
    y = at::rand({1024}, at::Device(at::kCUDA, 1));
    ASSERT_EQ(aoti_torch_cuda_graph_capture_end(graph_), AOTI_TORCH_SUCCESS);
  }
  // A replay runs on the current stream of the capture's device.
  c10::cuda::CUDAGuard device_guard(0);
  auto replay_stream = at::cuda::getStreamFromPool(false, 1);
  at::cuda::setCurrentCUDAStream(replay_stream);
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  replay_stream.synchronize();
  auto first = y.clone();
  ASSERT_EQ(replay(), AOTI_TORCH_SUCCESS);
  replay_stream.synchronize();
  EXPECT_FALSE(y.equal(first));
  at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream(1));
  ASSERT_EQ(aoti_torch_delete_cuda_graph(graph_), AOTI_TORCH_SUCCESS);
  graph_ = nullptr;
  EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool), AOTI_TORCH_SUCCESS);
}

TEST_F(AOTICudaGraphTest, ForeachCopyCopiesEachTensor) {
  auto longs = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);
  // Same dtype and layout, so the copies fuse, then mixed ones, which don't.
  for (bool fused : {true, false}) {
    std::vector<at::Tensor> dst = {
        at::empty({4}, at::kCUDA), at::empty({3, 2}, at::kCUDA)};
    std::vector<at::Tensor> src = {
        at::randn({4}, at::kCUDA), at::randn({3, 2}, at::kCUDA)};
    if (!fused) {
      dst[1] = at::empty({2, 3}, at::kCUDA).t();
      dst.push_back(at::empty({5}, longs));
      src.push_back(at::randint(100, {5}, longs));
    }
    std::vector<AtenTensorHandle> dst_handles, src_handles;
    for (size_t i = 0; i < dst.size(); ++i) {
      dst_handles.push_back(reinterpret_cast<AtenTensorHandle>(&dst[i]));
      src_handles.push_back(reinterpret_cast<AtenTensorHandle>(&src[i]));
    }
    auto n = static_cast<int64_t>(dst.size());
    ASSERT_EQ(
        aoti_torch_foreach_copy_(
            dst_handles.data(), n, src_handles.data(), n, fused),
        AOTI_TORCH_SUCCESS);
    for (size_t i = 0; i < dst.size(); ++i) {
      EXPECT_TRUE(dst[i].equal(src[i]));
    }
    EXPECT_EQ(
        aoti_torch_foreach_copy_(
            dst_handles.data(), n - 1, src_handles.data(), n, 0),
        AOTI_TORCH_FAILURE);
    dst[0] = at::empty({3}, at::kCUDA);
    EXPECT_EQ(
        aoti_torch_foreach_copy_(
            dst_handles.data(), n, src_handles.data(), n, 0),
        AOTI_TORCH_FAILURE);
  }
  EXPECT_EQ(
      aoti_torch_foreach_copy_(nullptr, 1, nullptr, 1, 0), AOTI_TORCH_FAILURE);
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
  CUDAGraphHandle graph = nullptr;
  EXPECT_EQ(aoti_torch_create_cuda_graph(0, -1, &graph), AOTI_TORCH_FAILURE);
  EXPECT_EQ(graph, nullptr);
  if (c10::cuda::device_count() < 2) {
    return;
  }
  // A pool on device 1 while device 0 is current.
  CUDAMemPoolHandle pool = nullptr;
  ASSERT_EQ(aoti_torch_create_cuda_mem_pool(1, &pool), AOTI_TORCH_SUCCESS);
  at::cuda::CUDAStreamGuard guard(*capture_stream_);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool, kThreadLocal),
      AOTI_TORCH_FAILURE);
  EXPECT_FALSE(stream_is_capturing());
  EXPECT_EQ(aoti_torch_delete_cuda_mem_pool(pool), AOTI_TORCH_SUCCESS);
}

// Run with PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync.
TEST_F(AOTICudaGraphTest, RejectsTheCudaMallocAsyncBackend) {
  // Split so hipify keeps the name, which ROCm's allocator reports too.
  if (c10::cuda::CUDACachingAllocator::name() !=
      "c"
      "udaMallocAsync") {
    GTEST_SKIP() << "Needs the cudaMallocAsync allocator backend";
  }
  at::cuda::CUDAStreamGuard guard(*capture_stream_);
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_begin(graph_, pool_, kThreadLocal),
      AOTI_TORCH_FAILURE);
  EXPECT_FALSE(stream_is_capturing());
}

TEST_F(AOTICudaGraphTest, EndingInvalidatedCaptureFailsButEndsIt) {
#ifdef USE_ROCM
  GTEST_SKIP() << "hipStreamEndCapture doesn't end an invalidated capture: "
                  "the stream rejects work until the next capture begins";
#endif
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
