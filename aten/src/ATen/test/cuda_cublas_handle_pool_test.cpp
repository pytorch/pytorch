#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <atomic>
#include <thread>
#include <vector>

#ifdef USE_ROCM
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>
#include <rocblas/rocblas.h>

#include <array>
#include <exception>
#endif

// Test concurrent access to getCurrentCUDABlasHandle and getCUDABlasLtWorkspace
// to verify that the data race fix is working correctly

TEST(CUDABlasHandlePoolTest, ConcurrentGetAndClearWorkspaces) {
  if (!at::cuda::is_available()) {
    return;
  }

  constexpr int num_accessor_threads = 15;
  constexpr int num_clear_threads = 5;
  constexpr int iterations_per_thread = 50;

  std::atomic<bool> stop{false};
  std::atomic<int> error_count{0};
  std::vector<std::thread> threads;
  threads.reserve(num_accessor_threads + num_clear_threads);

  // Launch accessor threads
  for (int i = 0; i < num_accessor_threads; ++i) {
    threads.emplace_back([&stop, &error_count]() {
      try {
        at::cuda::CUDAGuard device_guard(0);

        while (!stop.load(std::memory_order_relaxed)) {
          const auto handle = at::cuda::getCurrentCUDABlasHandle();
          const auto workspace = at::cuda::getCUDABlasLtWorkspace();

          if (handle == nullptr || workspace == nullptr) {
            error_count++;
          }
        }
      } catch (const std::exception&) {
        error_count++;
      }
    });
  }

  // Launch threads that clear workspaces
  for (int i = 0; i < num_clear_threads; ++i) {
    threads.emplace_back([&error_count]() {
      try {
        for (int j = 0; j < iterations_per_thread; ++j) {
          at::cuda::clearCublasWorkspaces();
          std::this_thread::yield();
        }
      } catch (const std::exception&) {
        error_count++;
      }
    });
  }

  // Let them run for a bit
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  stop.store(true, std::memory_order_relaxed);

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0);
}

#ifdef USE_ROCM

namespace {

// rocblas_is_user_managing_device_memory is False in every state and is useless
// as a predicate; this is the one that actually tracks the workspace binding.
bool rocblasOwnsWorkspace(cublasHandle_t handle) {
  return rocblas_is_managing_device_memory(
      reinterpret_cast<rocblas_handle>(handle));
}

size_t rocblasWorkspaceSize(cublasHandle_t handle) {
  size_t size = 0;
  rocblas_get_device_memory_size(
      reinterpret_cast<rocblas_handle>(handle), &size);
  return size;
}

} // namespace

// ATen binds its eager workspaces to an internal handle, so the public handle
// keeps the arena rocBLAS allocated at creation, as a cuBLAS handle keeps its
// default workspace.
TEST(CUDABlasHandlePoolTest, EagerWorkspaceLeavesPublicHandleArena) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }

  at::cuda::CUDAGuard device_guard(0);
  const auto public_handle = at::cuda::getCurrentCUDABlasHandle();
  ASSERT_TRUE(rocblasOwnsWorkspace(public_handle));
  const size_t arena_size = rocblasWorkspaceSize(public_handle);
  ASSERT_GT(arena_size, 0);

  cublasHandle_t internal_handle = nullptr;
  {
    auto scoped = at::cuda::getCurrentCUDABlasHandleWithWorkspace();
    internal_handle = scoped;
    EXPECT_NE(internal_handle, public_handle);
    EXPECT_FALSE(rocblasOwnsWorkspace(internal_handle));
  }

  // Unlike cuBLAS, rocblas_set_stream does not reset the workspace binding, so
  // the eager scope must unbind explicitly or it leaves the handle pointing at
  // memory that has gone back to the caching allocator.
  EXPECT_TRUE(rocblasOwnsWorkspace(internal_handle));
  EXPECT_EQ(rocblasWorkspaceSize(internal_handle), 0);
  EXPECT_EQ(at::cuda::getCurrentCUDABlasHandle(), public_handle);
  EXPECT_TRUE(rocblasOwnsWorkspace(public_handle));
  EXPECT_EQ(rocblasWorkspaceSize(public_handle), arena_size);
}

// A rocBLAS arena must not be used by two streams at once, and
// rocblas_set_stream does not wait for the old stream, so each stream gets its
// own public handle.
TEST(CUDABlasHandlePoolTest, EagerPublicHandleIsPerStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }

  at::cuda::CUDAGuard device_guard(0);
  const auto handle_on = [](c10::cuda::CUDAStream stream) {
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    return at::cuda::getCurrentCUDABlasHandle();
  };
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  const auto first_handle = handle_on(first);
  EXPECT_NE(handle_on(second), first_handle);
  EXPECT_EQ(handle_on(first), first_handle);
}

// The kernel Tensile picks for this shape uses the handle's workspace on gfx942
// and gfx950, so GEMMs from two streams that share one arena corrupt each
// other's results. Where the kernel uses no workspace the test cannot fail.
TEST(CUDABlasHandlePoolTest, EagerPublicHandleStreamsDoNotShareArena) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }

  at::cuda::CUDAGuard device_guard(0);
  constexpr int64_t m = 32;
  constexpr int64_t k = 65536;
  constexpr int64_t n = 2048;
  constexpr int iterations = 20;
  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  const std::array<at::Tensor, 2> inputs = {
      at::randn({m, k}, options), at::randn({m, k}, options) * 4};
  // With one weight shared by both streams the race usually leaves results
  // intact, so each stream gets its own.
  const std::array<at::Tensor, 2> weights = {
      at::randn({n, k}, options), at::randn({n, k}, options)};
  const std::array<c10::cuda::CUDAStream, 2> streams = {
      c10::cuda::getStreamFromPool(), c10::cuda::getStreamFromPool()};

  const auto gemm = [&](size_t s, const at::Tensor& out) {
    const float alpha = 1;
    const float beta = 0;
    const auto handle =
        reinterpret_cast<rocblas_handle>(at::cuda::getCurrentCUDABlasHandle());
    return rocblas_gemm_ex(
        handle,
        rocblas_operation_transpose,
        rocblas_operation_none,
        n,
        m,
        k,
        &alpha,
        weights[s].data_ptr(),
        rocblas_datatype_bf16_r,
        k,
        inputs[s].data_ptr(),
        rocblas_datatype_bf16_r,
        k,
        &beta,
        out.data_ptr(),
        rocblas_datatype_bf16_r,
        n,
        out.data_ptr(),
        rocblas_datatype_bf16_r,
        n,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0,
        0);
  };

  std::array<at::Tensor, 2> expected;
  for (size_t s = 0; s < streams.size(); ++s) {
    expected[s] = at::empty({m, n}, options);
    ASSERT_EQ(gemm(s, expected[s]), rocblas_status_success);
  }
  std::vector<std::array<at::Tensor, 2>> outputs(iterations);
  for (auto& output : outputs) {
    output = {at::empty({m, n}, options), at::empty({m, n}, options)};
  }
  at::cuda::device_synchronize();

  // Stall both streams so the GEMMs queue up behind the stall and then run
  // concurrently.
  for (const auto& stream : streams) {
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    at::cuda::sleep(50'000'000);
  }
  for (const auto& output : outputs) {
    for (size_t s = 0; s < streams.size(); ++s) {
      c10::cuda::CUDAStreamGuard stream_guard(streams[s]);
      ASSERT_EQ(gemm(s, output[s]), rocblas_status_success);
    }
  }
  at::cuda::device_synchronize();

  int wrong = 0;
  for (const auto& output : outputs) {
    for (size_t s = 0; s < streams.size(); ++s) {
      const float scale = expected[s].abs().max().item<float>();
      const float error =
          (output[s].to(at::kFloat) - expected[s].to(at::kFloat))
              .abs()
              .max()
              .item<float>();
      wrong += error > 0.1f * scale;
    }
  }
  EXPECT_EQ(wrong, 0) << "of " << 2 * iterations << " outputs";
}

TEST(CUDABlasHandlePoolTest, CachedWorkspaceLeavesHandleUserOwned) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (!at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires TORCH_CUBLAS_WORKSPACE_CACHE=1";
  }

  at::cuda::CUDAGuard device_guard(0);
  auto scoped = at::cuda::getCurrentCUDABlasHandleWithWorkspace();
  EXPECT_FALSE(rocblasOwnsWorkspace(scoped));
}

// rocblas_set_workspace frees whatever the handle currently manages, and a
// handle owns an arena from creation, so the first bind performs a free. A free
// is illegal under stream capture, which is why internal handles drop that
// arena when they are created. Handles are thread local, so the capture must
// run on a thread that has not issued a gemm yet or the hazard is already spent
// and the test is vacuous.
TEST(CUDABlasHandlePoolTest, EagerWorkspaceBindIsCaptureSafe) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }

  // hipBLASLt passes the workspace per call and never binds one to the rocBLAS
  // handle, so the legacy backend is the only one that exercises this.
  const auto prev_backend = at::globalContext().blasPreferredBackend();
  at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublas);

  // Only split-K / stream-K kernels touch the workspace on gfx9, and those need
  // a large K. A square shape writes zero workspace bytes. Tensile picks per
  // arch, so this K was measured on gfx942 rather than carried over.
  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  const auto a = at::randn({32, 65536}, options);
  const auto b = at::randn({65536, 2048}, options);
  const auto expected = at::mm(a, b);
  at::cuda::device_synchronize();

  at::Tensor captured;
  std::exception_ptr failure;
  std::thread worker([&]() {
    try {
      at::cuda::CUDAGuard device_guard(0);
      auto stream = c10::cuda::getStreamFromPool();
      c10::cuda::CUDAStreamGuard stream_guard(stream);
      // Creates this thread's handles before capture starts; creating a
      // handle is illegal under capture.
      (void)at::cuda::getCurrentCUDABlasHandle();

      at::cuda::CUDAGraph graph;
      graph.capture_begin();
      captured = at::mm(a, b);
      graph.capture_end();
      graph.replay();
      stream.synchronize();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  worker.join();

  at::globalContext().setBlasPreferredBackend(prev_backend);
  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(at::allclose(captured, expected, 1e-2, 2e-1));
}

#endif // USE_ROCM

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(1);
  return RUN_ALL_TESTS();
}
