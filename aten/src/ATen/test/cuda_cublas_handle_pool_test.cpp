#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>

#include <atomic>
#include <thread>
#include <vector>

#if defined(__linux__) && !defined(USE_ROCM)
#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#endif

#ifdef USE_ROCM
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/util/env.h>
#include <rocblas/rocblas.h>

#include <array>
#include <chrono>
#include <exception>
#include <future>
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

#if defined(__linux__) && !defined(USE_ROCM)
namespace {
// A negative count disables failure injection for the other tests.
thread_local int create_attempts = -1;
} // namespace

// Override the dynamically linked entry point to force an allocation failure.
extern "C" C10_EXPORT cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  if (create_attempts >= 0) {
    static thread_local std::int64_t reserved_before_create = 0;
    constexpr auto aggregate =
        static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
        c10::cuda::current_device());
    const auto reserved = stats.reserved_bytes[aggregate].current;
    if (++create_attempts == 1) {
      EXPECT_GT(reserved, 0);
      reserved_before_create = reserved;
      return CUBLAS_STATUS_ALLOC_FAILED;
    }
    // Verify PyTorch released cached memory before the cuBLAS retry.
    EXPECT_LT(reserved, reserved_before_create);
  }
  static const auto real_create = reinterpret_cast<decltype(&cublasCreate_v2)>(
      dlsym(RTLD_NEXT, "cublasCreate_v2"));
  TORCH_CHECK(real_create, "Cannot find the real cublasCreate_v2");
  return real_create(handle);
}

TEST(CUDABlasHandlePoolDeathTest, RetriesAllocationFailure) {
  if (!at::cuda::is_available() ||
      !c10::cuda::CUDACachingAllocator::isEnabled()) {
    GTEST_SKIP() << "requires a CUDA device and allocator caching";
  }
  // Re-exec to get a fresh handle pool without inheriting a CUDA context.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        at::globalContext().lazyInitDevice(at::kCUDA);
        auto* cached = c10::cuda::CUDACachingAllocator::raw_alloc(1024 * 1024);
        c10::cuda::CUDACachingAllocator::raw_delete(cached);
        C10_CUDA_CHECK(cudaDeviceSynchronize());
        create_attempts = 0;
        EXPECT_NE(at::cuda::getCurrentCUDABlasHandle(/*setup=*/false), nullptr);
        EXPECT_EQ(create_attempts, 2);
        // Propagate child assertion failures to the parent.
        std::_Exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS),
      "");
}
#endif // defined(__linux__) && !defined(USE_ROCM)

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

// An exiting thread returns its stream-keyed handles while work it queued may
// still be running, and such a handle cannot serve two streams at once, so the
// stream that takes it next must wait for that work.
template <typename GetHandle>
void expectHandleFromExitedThreadWaitsForItsStream(
    GetHandle get_handle,
    bool wrap_first = false) {
  at::cuda::CUDAGuard device_guard(0);
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  decltype(get_handle()) returned = nullptr;
  at::cuda::CUDAEvent owner_done;
  std::thread owner([&] {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(
        wrap_first ? c10::cuda::getStreamFromExternal(first.stream(), 0)
                   : first);
    returned = get_handle();
    // Stands in for work still queued when the thread exits.
    at::cuda::sleep(200'000'000);
    owner_done.record(first);
  });
  owner.join();

  decltype(get_handle()) reused = nullptr;
  at::cuda::CUDAEvent after_reuse;
  std::thread next_owner([&] {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(second);
    reused = get_handle();
    after_reuse.record(second);
  });
  next_owner.join();

  // The pool hands out the handle returned last.
  ASSERT_EQ(reused, returned);
  // The reuse finishes only after the exited thread's work.
  after_reuse.synchronize();
  EXPECT_TRUE(owner_done.query());
}

// A hipBLASLt handle owns the synchronizer its split-K kernels spin on.
TEST(CUDABlasHandlePoolTest, LtHandleFromExitedThreadWaitsForItsStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  expectHandleFromExitedThreadWaitsForItsStream(
      [] { return at::cuda::getCurrentCUDABlasLtHandle(); });
}

// AOTInductor runs ATen ops on the stream it is given, wrapped by
// getStreamFromExternal, so the pool must recognize its own streams by address.
TEST(
    CUDABlasHandlePoolTest,
    LtHandleFromExitedThreadWaitsForItsWrappedPoolStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  expectHandleFromExitedThreadWaitsForItsStream(
      [] { return at::cuda::getCurrentCUDABlasLtHandle(); },
      /*wrap_first=*/true);
}

// A thread that first needs a handle mid-capture, like an autograd worker in
// backward, takes one from the pool on a capturing stream. That stream cannot
// wait on work outside its capture, and its kernels run whenever the graph is
// replayed, so the host waits for the exited thread's work instead. The capture
// begins directly because CUDAGraph::capture_begin() reserves a handle for the
// capturing thread before capture starts.
TEST(
    CUDABlasHandlePoolTest,
    LtHandleFromExitedThreadWaitsForItsStreamUnderCapture) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  cublasLtHandle_t returned = nullptr;
  at::cuda::CUDAEvent owner_done;
  std::thread owner([&] {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(first);
    returned = at::cuda::getCurrentCUDABlasLtHandle();
    at::cuda::sleep(200'000'000);
    owner_done.record(first);
  });
  owner.join();

  ASSERT_EQ(
      cudaStreamBeginCapture(second, cudaStreamCaptureModeGlobal), cudaSuccess);
  cublasLtHandle_t reused = nullptr;
  bool owner_done_at_reuse = false;
  std::exception_ptr failure;
  std::thread worker([&] {
    try {
      at::cuda::CUDAGuard guard(0);
      c10::cuda::CUDAStreamGuard stream_guard(second);
      reused = at::cuda::getCurrentCUDABlasLtHandle();
      // Global capture mode forbids querying an event outside the capture.
      c10::cuda::CUDAStreamCaptureModeGuard mode_guard(
          cudaStreamCaptureModeRelaxed);
      owner_done_at_reuse = owner_done.query();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  worker.join();
  cudaGraph_t graph = nullptr;
  const auto end_status = cudaStreamEndCapture(second, &graph);
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }

  if (failure) {
    std::rethrow_exception(failure);
  }
  EXPECT_EQ(end_status, cudaSuccess) << cudaGetErrorString(end_status);
  ASSERT_EQ(reused, returned);
  EXPECT_TRUE(owner_done_at_reuse);
}

// The host waits for the exited thread's work without holding the pool lock,
// since that work may wait on a thread that has yet to take a handle. Here it
// waits for a value written only once another thread has taken one. Which of
// the two threads takes the exited thread's handle is a race, so the test
// retries until the waiting thread does.
TEST(CUDABlasHandlePoolTest, LtHandleHostWaitLetsOtherThreadsTakeHandles) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  int can_wait = 0;
  ASSERT_EQ(
      hipDeviceGetAttribute(
          &can_wait, hipDeviceAttributeCanUseStreamWaitValue, 0),
      hipSuccess);
  if (can_wait == 0) {
    GTEST_SKIP() << "requires hipStreamWaitValue64";
  }
  void* gate = nullptr;
  ASSERT_EQ(
      hipExtMallocWithFlags(&gate, sizeof(uint64_t), hipMallocSignalMemory),
      hipSuccess);
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  const auto third = c10::cuda::getStreamFromPool();
  const auto opener = c10::cuda::getStreamFromPool();
  constexpr uint64_t kAttempts = 5;
  // No stream may still be waiting on the gate when it is freed.
  auto free_gate = c10::make_scope_exit([&] {
    (void)hipStreamWriteValue64(opener, gate, kAttempts, 0);
    (void)hipStreamSynchronize(first);
    (void)hipFree(gate);
  });
  ASSERT_EQ(hipStreamWriteValue64(first, gate, 0, 0), hipSuccess);
  // Attempt k gates the exited thread's work on the value k.
  for (uint64_t attempt = 1; attempt <= kAttempts; ++attempt) {
    // Creating a handle fails while a stream is capturing, so the other thread
    // takes a spare, which the pool hands out after the exited thread's handle.
    at::cuda::ensureCublasLtHandlesAvailable(2);
    cublasLtHandle_t returned = nullptr;
    std::thread owner([&] {
      at::cuda::CUDAGuard guard(0);
      c10::cuda::CUDAStreamGuard stream_guard(first);
      returned = at::cuda::getCurrentCUDABlasLtHandle();
      EXPECT_EQ(
          hipStreamWaitValue64(first, gate, attempt, hipStreamWaitValueGte),
          hipSuccess);
    });
    owner.join();

    ASSERT_EQ(
        cudaStreamBeginCapture(second, cudaStreamCaptureModeRelaxed),
        cudaSuccess);
    std::atomic<bool> reserving{false};
    cublasLtHandle_t reused = nullptr;
    std::exception_ptr failure;
    std::thread waiter([&] {
      reserving = true;
      try {
        at::cuda::CUDAGuard guard(0);
        c10::cuda::CUDAStreamGuard stream_guard(second);
        reused = at::cuda::getCurrentCUDABlasLtHandle();
      } catch (...) {
        failure = std::current_exception();
      }
    });
    while (!reserving) {
      std::this_thread::yield();
    }
    auto other = std::async(std::launch::async, [&] {
      at::cuda::CUDAGuard guard(0);
      c10::cuda::CUDAStreamGuard stream_guard(third);
      return at::cuda::getCurrentCUDABlasLtHandle();
    });
    const bool other_took_handle =
        other.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
    // Lets the exited thread's work finish either way.
    EXPECT_EQ(hipStreamWriteValue64(opener, gate, attempt, 0), hipSuccess);
    waiter.join();
    first.synchronize();
    cudaGraph_t graph = nullptr;
    const auto end_status = cudaStreamEndCapture(second, &graph);
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    (void)other.get();

    if (failure) {
      std::rethrow_exception(failure);
    }
    EXPECT_EQ(end_status, cudaSuccess) << cudaGetErrorString(end_status);
    if (reused == returned) {
      EXPECT_TRUE(other_took_handle);
      return;
    }
  }
  FAIL() << "the other thread took the exited thread's handle first in all "
         << kAttempts << " attempts";
}

// The owner of an external stream may destroy it before the thread that used
// it exits, and HIP crashes on a destroyed stream, so the pool must leave that
// stream alone when the thread returns its handle.
TEST(CUDABlasHandlePoolTest, LtHandleReturnSkipsDestroyedExternalStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  cublasLtHandle_t returned = nullptr;
  std::thread owner([&] {
    at::cuda::CUDAGuard guard(0);
    cudaStream_t external = nullptr;
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&external, cudaStreamNonBlocking),
        cudaSuccess);
    {
      c10::cuda::CUDAStreamGuard stream_guard(
          c10::cuda::getStreamFromExternal(external, 0));
      returned = at::cuda::getCurrentCUDABlasLtHandle();
    }
    EXPECT_EQ(cudaStreamDestroy(external), cudaSuccess);
  });
  owner.join();

  cublasLtHandle_t reused = nullptr;
  std::thread next_owner([&] {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromPool());
    reused = at::cuda::getCurrentCUDABlasLtHandle();
  });
  next_owner.join();
  // The handle still goes back to the pool.
  EXPECT_EQ(reused, returned);
}

// A public rocBLAS handle keeps the workspace bound at its creation.
TEST(CUDABlasHandlePoolTest, PublicHandleFromExitedThreadWaitsForItsStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }
  expectHandleFromExitedThreadWaitsForItsStream(
      [] { return at::cuda::getCurrentCUDABlasHandle(); });
}

// With the check in fence mode, which main() turns on, a launch on a stream
// other than the one the handle last used waits for that stream's queued work.
TEST(CUDABlasHandlePoolTest, LtHandleStreamCheckOrdersCrossStreamLaunch) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  // The check never dereferences the handle, so any distinct pointer works.
  int key = 0;
  const auto handle = reinterpret_cast<cublasLtHandle_t>(&key);
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  at::cuda::CUDAEvent first_done;
  {
    c10::cuda::CUDAStreamGuard stream_guard(first);
    at::cuda::checkCublasLtHandleStream(handle, first);
    at::cuda::sleep(200'000'000);
    first_done.record(first);
  }

  at::cuda::CUDAEvent after_launch;
  {
    c10::cuda::CUDAStreamGuard stream_guard(second);
    at::cuda::checkCublasLtHandleStream(handle, second);
    after_launch.record(second);
  }

  // The launch finishes only after the work queued before it.
  after_launch.synchronize();
  EXPECT_TRUE(first_done.query());
}

// The same when the handle last ran on another device: from a pool stream to
// the other device's default stream, then between the two devices' default
// streams, which are both the null stream.
TEST(CUDABlasHandlePoolTest, LtHandleStreamCheckOrdersCrossDeviceLaunch) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::device_count() < 2) {
    GTEST_SKIP() << "requires two GPUs";
  }
  int key = 0;
  const auto handle = reinterpret_cast<cublasLtHandle_t>(&key);
  const auto pool_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false, 0);
  const auto default0 = c10::cuda::getDefaultCUDAStream(0);
  const auto default1 = c10::cuda::getDefaultCUDAStream(1);
  // A stream's first launch can take longer than the sleeps below, so the
  // default streams launch here first.
  for (const auto& stream : {default0, default1}) {
    at::cuda::CUDAEvent ready;
    ready.record(stream);
    ready.synchronize();
  }
  const auto expect_ordered = [&](c10::cuda::CUDAStream last,
                                  c10::cuda::CUDAStream next) {
    at::cuda::CUDAEvent last_done;
    {
      c10::cuda::CUDAStreamGuard stream_guard(last);
      at::cuda::sleep(200'000'000);
      last_done.record(last);
    }
    c10::cuda::CUDAStreamGuard stream_guard(next);
    at::cuda::checkCublasLtHandleStream(handle, next);
    at::cuda::CUDAEvent after_launch;
    after_launch.record(next);
    after_launch.synchronize();
    EXPECT_TRUE(last_done.query())
        << "launch on device " << static_cast<int>(next.device_index());
  };
  {
    c10::cuda::CUDAStreamGuard stream_guard(pool_stream);
    at::cuda::checkCublasLtHandleStream(handle, pool_stream);
  }

  expect_ordered(pool_stream, default1);
  expect_ordered(default1, default0);
}

// An SM carveout launch runs on a stream other than the current one, which the
// pool holds the handle for: the carveout stream waits for the current one
// before the launch, and the current one waits for it after. So the check
// tracks the current stream, and orders the launch stream itself.
TEST(CUDABlasHandlePoolTest, LtHandleStreamCheckOrdersCarveoutLaunch) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  int key = 0;
  const auto handle = reinterpret_cast<cublasLtHandle_t>(&key);
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  const auto carveout = c10::cuda::getStreamFromPool();
  at::cuda::CUDAEvent first_done;
  {
    c10::cuda::CUDAStreamGuard stream_guard(first);
    at::cuda::checkCublasLtHandleStream(handle, carveout);
    // Stands in for the work of `first`, which waits for the launch.
    at::cuda::sleep(200'000'000);
    first_done.record(first);
  }

  at::cuda::CUDAEvent after_launch;
  {
    c10::cuda::CUDAStreamGuard stream_guard(second);
    at::cuda::checkCublasLtHandleStream(handle, carveout);
    after_launch.record(carveout);
  }

  after_launch.synchronize();
  EXPECT_TRUE(first_done.query());
}

// The pool cannot fence a handle whose stream was capturing when its thread
// exited, so the check orders the next owner's launch itself, here after work
// queued on that stream once the capture ended.
TEST(CUDABlasHandlePoolTest, LtHandleStreamCheckOrdersUnfencedHandover) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  cublasLtHandle_t returned = nullptr;
  std::promise<void> launched;
  std::promise<void> capturing;
  std::thread owner([&] {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(first);
    returned = at::cuda::getCurrentCUDABlasLtHandle();
    at::cuda::checkCublasLtHandleStream(returned, first);
    launched.set_value();
    capturing.get_future().wait();
  });
  launched.get_future().wait();
  const auto begin_status =
      cudaStreamBeginCapture(first, cudaStreamCaptureModeRelaxed);
  capturing.set_value();
  owner.join();
  cudaGraph_t graph = nullptr;
  const auto end_status = begin_status == cudaSuccess
      ? cudaStreamEndCapture(first, &graph)
      : begin_status;
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  ASSERT_EQ(end_status, cudaSuccess) << cudaGetErrorString(end_status);
  at::cuda::CUDAEvent first_done;
  {
    c10::cuda::CUDAStreamGuard stream_guard(first);
    // Stands in for the captured launches, replayed on that stream.
    at::cuda::sleep(200'000'000);
    first_done.record(first);
  }

  cublasLtHandle_t reused = nullptr;
  at::cuda::CUDAEvent after_launch;
  std::thread next_owner([&] {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(second);
    reused = at::cuda::getCurrentCUDABlasLtHandle();
    at::cuda::checkCublasLtHandleStream(reused, second);
    after_launch.record(second);
  });
  next_owner.join();

  ASSERT_EQ(reused, returned);
  after_launch.synchronize();
  EXPECT_TRUE(first_done.query());
}

#endif // USE_ROCM

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
#ifdef USE_ROCM
  // The hipBLASLt stream check reads its mode once, at the first launch, so
  // fence mode for its tests is set before any test runs.
  c10::utils::set_env("TORCH_HIPBLASLT_CHECK_HANDLE_STREAMS", "fence");
#endif
  // Every TEST() below already guards on at::cuda::is_available() and returns
  // early when there's no GPU. init() below is not: it unconditionally touches
  // the CUDA driver (DeviceCachingAllocator's ctor calls cudaGetDeviceProperties),
  // which aborts the whole process on a GPU-less machine before any TEST() body
  // gets a chance to self-skip.
  if (at::cuda::is_available()) {
    c10::cuda::CUDACachingAllocator::init(1);
  }
  return RUN_ALL_TESTS();
}
