#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>

#include <atomic>
#include <thread>
#include <vector>

#ifdef USE_ROCM
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
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

// ATen binds its eager workspaces to an internal handle. The public handle is
// bound at creation to a caching-allocator buffer that replaces the arena
// rocBLAS allocated, so its workspace is counted by the allocator.
TEST(CUDABlasHandlePoolTest, EagerWorkspaceBindsPublicHandleToAllocator) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }

  at::cuda::CUDAGuard device_guard(0);
  const auto public_handle = at::cuda::getCurrentCUDABlasHandle();
  EXPECT_FALSE(rocblasOwnsWorkspace(public_handle));
  const size_t bound_size = rocblasWorkspaceSize(public_handle);
  EXPECT_GE(bound_size, at::cuda::getChosenWorkspaceSize());

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
  EXPECT_FALSE(rocblasOwnsWorkspace(public_handle));
  EXPECT_EQ(rocblasWorkspaceSize(public_handle), bound_size);
}

// ATen binds its workspaces to separate handles, so a workspace the caller
// binds to the public handle stays bound across ATen operations.
TEST(CUDABlasHandlePoolTest, EagerCallerWorkspaceSurvivesAtenOps) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }

  // The legacy backend is the one whose operations bind a rocBLAS workspace.
  const auto prev_backend = at::globalContext().blasPreferredBackend();
  at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublas);
  auto restore_backend = c10::make_scope_exit(
      [&] { at::globalContext().setBlasPreferredBackend(prev_backend); });

  at::cuda::CUDAGuard device_guard(0);
  // Unbinding at the end leaves this stream's public handle without a workspace,
  // so the test uses a high-priority stream, which no other test here does.
  c10::cuda::CUDAStreamGuard stream_guard(
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true));
  const auto handle = at::cuda::getCurrentCUDABlasHandle();
  const auto rocblas = reinterpret_cast<rocblas_handle>(handle);
  constexpr int64_t size = 32 * 1024 * 1024;
  const auto workspace = at::empty(
      {size}, at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
  EXPECT_EQ(
      rocblas_set_workspace(rocblas, workspace.data_ptr(), size),
      rocblas_status_success);

  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  (void)at::mm(at::randn({64, 64}, options), at::randn({64, 64}, options));
  EXPECT_EQ(at::cuda::getCurrentCUDABlasHandle(), handle);
  EXPECT_FALSE(rocblasOwnsWorkspace(handle));
  EXPECT_EQ(rocblasWorkspaceSize(handle), static_cast<size_t>(size));

  EXPECT_EQ(rocblas_set_workspace(rocblas, nullptr, 0), rocblas_status_success);
}

// A rocBLAS workspace must not be used by two streams at once, and
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
// and gfx950, so GEMMs from two streams that share one workspace corrupt each
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
// arena when they are created. The capture runs on a new thread so that the
// thread's internal handle is created by the getCurrentCUDABlasHandle() call
// below and first bound inside the capture. On a thread that had already run a
// GEMM, both would have happened earlier and the test would check neither.
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
  auto restore_backend = c10::make_scope_exit(
      [&] { at::globalContext().setBlasPreferredBackend(prev_backend); });

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

  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_TRUE(at::allclose(captured, expected, 1e-2, 2e-1));
}

// One capture handle serves all of a thread's capturing streams, so it must
// switch pointer modes with them.
TEST(CUDABlasHandlePoolTest, CaptureHandleKeepsPointerModePerStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }
  at::cuda::CUDAGuard device_guard(0);
  auto s1 = c10::cuda::getStreamFromPool();
  auto s2 = c10::cuda::getStreamFromPool();
  auto t = at::zeros({1}, at::TensorOptions().device(at::kCUDA));
  const auto mode = [](cublasHandle_t handle) {
    rocblas_pointer_mode m = rocblas_pointer_mode_host;
    EXPECT_EQ(rocblas_get_pointer_mode(reinterpret_cast<rocblas_handle>(handle), &m), rocblas_status_success);
    return m;
  };
  // Handles cannot be created under capture.
  {
    c10::cuda::CUDAStreamGuard guard(s2);
    (void)at::cuda::getCurrentCUDABlasHandle();
  }
  c10::cuda::CUDAStreamGuard stream_guard(s1);
  (void)at::cuda::getCurrentCUDABlasHandle();
  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  t.add_(1);
  auto h1 = reinterpret_cast<rocblas_handle>(at::cuda::getCurrentCUDABlasHandle());
  EXPECT_EQ(rocblas_set_pointer_mode(h1, rocblas_pointer_mode_device), rocblas_status_success);
  at::cuda::CUDAEvent fork, join;
  fork.record(s1);
  fork.block(s2);
  {
    c10::cuda::CUDAStreamGuard guard(s2);
    EXPECT_EQ(mode(at::cuda::getCurrentCUDABlasHandle()), rocblas_pointer_mode_host);
    join.record(s2);
  }
  join.block(s1);
  EXPECT_EQ(mode(at::cuda::getCurrentCUDABlasHandle()), rocblas_pointer_mode_device);
  EXPECT_EQ(rocblas_set_pointer_mode(h1, rocblas_pointer_mode_host), rocblas_status_success);
  graph.capture_end();
}

#endif // USE_ROCM

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(1);
  return RUN_ALL_TESTS();
}
