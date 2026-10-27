// Checks the CUDA graph capture functions of the C shim.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <optional>

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

  // Captures fn's work on the capture stream and returns its result.
  template <typename Fn>
  at::Tensor capture(Fn fn) {
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_begin(
            graph_, pool_, capture_stream_->stream()),
        AOTI_TORCH_SUCCESS);
    at::Tensor out;
    {
      at::cuda::CUDAStreamGuard guard(*capture_stream_);
      out = fn();
    }
    EXPECT_EQ(
        aoti_torch_cuda_graph_capture_end(graph_, /*instantiate=*/true),
        AOTI_TORCH_SUCCESS);
    return out;
  }

  bool stream_is_capturing() {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    EXPECT_EQ(
        cudaStreamIsCapturing(capture_stream_->stream(), &status), cudaSuccess);
    return status != cudaStreamCaptureStatusNone;
  }

  // Checks that the stream and the allocator are left fit for a new capture.
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

  AOTITorchError replay() {
    return aoti_torch_cuda_graph_replay(
        graph_, at::cuda::getCurrentCUDAStream().stream());
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

TEST_F(AOTICudaGraphTest, DiscardedCaptureCannotReplay) {
  auto x = at::zeros({8}, at::kCUDA);
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  {
    at::cuda::CUDAStreamGuard guard(*capture_stream_);
    x.add_(1);
  }
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_end(graph_, /*instantiate=*/false),
      AOTI_TORCH_SUCCESS);

  EXPECT_FALSE(stream_is_capturing());
  EXPECT_EQ(replay(), AOTI_TORCH_FAILURE);
}

TEST_F(AOTICudaGraphTest, DiscardingInvalidatedCaptureSucceeds) {
  ASSERT_EQ(
      aoti_torch_cuda_graph_capture_begin(
          graph_, pool_, capture_stream_->stream()),
      AOTI_TORCH_SUCCESS);
  // Synchronizing a capturing stream is illegal and invalidates the capture.
  EXPECT_NE(cudaStreamSynchronize(capture_stream_->stream()), cudaSuccess);
  (void)cudaGetLastError();
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_end(graph_, /*instantiate=*/false),
      AOTI_TORCH_SUCCESS);
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
  EXPECT_EQ(
      aoti_torch_cuda_graph_capture_end(graph_, /*instantiate=*/true),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(replay(), AOTI_TORCH_FAILURE);
}
