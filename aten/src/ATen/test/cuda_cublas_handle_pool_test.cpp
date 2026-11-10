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
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <rocblas/rocblas.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
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
  // Unbinding at the end leaves this stream's public handle without a
  // workspace, so the test uses a high-priority stream, on which no other test
  // here uses a public handle.
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
// arena when they are created. The capture runs on a new thread, which takes
// its internal handle for captures in the getCurrentCUDABlasHandle() call
// below: a new handle, or, when earlier tests ran in this process, one that
// one of their threads returned. Either way, this thread binds it first inside
// the capture.
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

// Runs `body` on a new thread and rethrows what it throws, so that a failure
// fails the test instead of terminating the binary.
template <typename Body>
void runOnThread(Body body) {
  std::exception_ptr failure;
  std::thread thread([&] {
    try {
      body();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  thread.join();
  if (failure) {
    std::rethrow_exception(failure);
  }
}

namespace {

// Lets a test hold up the creation of FakeHandlePool handles, to see what other
// threads can do while a handle is being created.
struct FakeHandleCreation {
  std::mutex mutex;
  std::condition_variable changed;
  bool hold = false;
  int waiting = 0;
};

FakeHandleCreation& fakeHandleCreation() {
  static FakeHandleCreation creation;
  return creation;
}

void createFakeHandle(int** handle) {
  auto& creation = fakeHandleCreation();
  std::unique_lock<std::mutex> lock(creation.mutex);
  ++creation.waiting;
  creation.changed.notify_all();
  creation.changed.wait(lock, [&] { return !creation.hold; });
  --creation.waiting;
  *handle = new int(0);
}

void destroyFakeHandle(int* handle) {
  delete handle;
}

using FakeHandlePool =
    at::cuda::DeviceThreadHandlePool<int*, createFakeHandle, destroyFakeHandle>;

// Reserves a handle through a window that is destroyed on return, as when the
// reserving thread exits. The pool uses streams only as keys.
int* reserveAndExit(FakeHandlePool& pool, int device, void* stream) {
  std::unique_ptr<FakeHandlePool::PoolWindow> window(pool.newPoolWindow());
  return window->reserve(device, stream);
}

// Set once a test has kept a hipStreamPerThread handle. The warning about that
// prints once for each copy of warnPerThreadStreamHandleKept(), and where this
// binary shares ATen's copy, as when linked statically, a FakeHandlePool prints
// it too, so later tests in the same process do not see it.
bool kept_per_thread_stream_handle = false;

// Creating a handle can synchronize the device, and a pool's lock is shared by
// threads on every device. While one thread is held creating a handle on
// device 0 through `create`, a thread on device 1 must still take the handle an
// exited thread returned for its stream.
template <typename Create>
void expectHeldCreationLeavesPoolUnlocked(Create create) {
  auto pool = std::make_shared<FakeHandlePool>();
  void* const returned_stream = reinterpret_cast<void*>(0x10);
  int* const returned = reserveAndExit(*pool, 1, returned_stream);

  auto& creation = fakeHandleCreation();
  {
    std::lock_guard<std::mutex> lock(creation.mutex);
    creation.hold = true;
  }
  std::exception_ptr creator_failure;
  std::thread creator([&] {
    try {
      create(*pool);
    } catch (...) {
      creator_failure = std::current_exception();
    }
  });
  const auto finish_creation = [&] {
    {
      std::lock_guard<std::mutex> lock(creation.mutex);
      creation.hold = false;
    }
    creation.changed.notify_all();
    if (creator.joinable()) {
      creator.join();
    }
  };
  auto finish_on_exit = c10::make_scope_exit(finish_creation);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  {
    std::unique_lock<std::mutex> lock(creation.mutex);
    ASSERT_TRUE(creation.changed.wait_until(
        lock, deadline, [&] { return creation.waiting == 1; }));
  }

  // While the creation is held, a thread on another device takes the
  // returned handle.
  std::promise<int*> taken;
  auto taken_future = taken.get_future();
  std::thread taker([&] {
    try {
      taken.set_value(reserveAndExit(*pool, 1, returned_stream));
    } catch (...) {
      taken.set_exception(std::current_exception());
    }
  });
  const bool taken_in_time =
      taken_future.wait_until(deadline) == std::future_status::ready;
  finish_creation();
  taker.join();
  if (creator_failure) {
    std::rethrow_exception(creator_failure);
  }

  EXPECT_TRUE(taken_in_time);
  EXPECT_EQ(taken_future.get(), returned);
}

} // namespace

// A stream-keyed handle is created without the pool's lock.
TEST(CUDABlasHandlePoolTest, StreamHandleCreationLeavesPoolUnlocked) {
  expectHeldCreationLeavesPoolUnlocked([](FakeHandlePool& pool) {
    (void)reserveAndExit(pool, 0, reinterpret_cast<void*>(0x20));
  });
}

// So is a spare, such as the ones graph capture stocks.
TEST(CUDABlasHandlePoolTest, SpareCreationLeavesPoolUnlocked) {
  expectHeldCreationLeavesPoolUnlocked(
      [](FakeHandlePool& pool) { pool.ensureAvailable(0, 1); });
}

// A stream takes a handle an exited thread returned for it before a spare,
// which no stream has used, so that the spares graph capture stocks are left
// for threads that first need a handle mid-capture.
TEST(CUDABlasHandlePoolTest, StreamHandleFromExitedThreadIsTakenBeforeSpare) {
  auto pool = std::make_shared<FakeHandlePool>();
  void* const stream = reinterpret_cast<void*>(0x10);
  void* const other_stream = reinterpret_cast<void*>(0x20);
  int* const returned = reserveAndExit(*pool, 0, stream);
  pool->ensureAvailable(0, 1);
  int* const spare = pool->available_handles[0].back();

  std::unique_ptr<FakeHandlePool::PoolWindow> window(pool->newPoolWindow());
  EXPECT_EQ(window->reserve(0, stream), returned);
  EXPECT_EQ(window->reserve(0, other_stream), spare);
}

// A handle an exited thread returned for one stream is not given to another
// stream, even when the pool has no other handle. The tests below check this
// for each of ATen's pools on a GPU; this one needs none.
TEST(CUDABlasHandlePoolTest, StreamHandleFromExitedThreadStaysOnItsStream) {
  auto pool = std::make_shared<FakeHandlePool>();
  void* const stream = reinterpret_cast<void*>(0x10);
  void* const other_stream = reinterpret_cast<void*>(0x20);
  int* const returned = reserveAndExit(*pool, 0, stream);
  EXPECT_NE(reserveAndExit(*pool, 0, other_stream), returned);
  EXPECT_EQ(reserveAndExit(*pool, 0, stream), returned);
}

// An exiting thread returns its stream-keyed handles while work it queued may
// still be running, and such a handle cannot serve two streams at once, so the
// pool hands it out again only for the stream it ran on.
template <typename GetHandle>
void expectHandleFromExitedThreadStaysOnItsStream(
    GetHandle get_handle,
    c10::cuda::CUDAStream stream) {
  at::cuda::CUDAGuard device_guard(0);
  const auto handle_on = [&](c10::cuda::CUDAStream s) {
    decltype(get_handle()) handle = nullptr;
    runOnThread([&] {
      at::cuda::CUDAGuard guard(0);
      c10::cuda::CUDAStreamGuard stream_guard(s);
      handle = get_handle();
    });
    return handle;
  };
  const auto returned = handle_on(stream);
  EXPECT_NE(handle_on(c10::cuda::getStreamFromPool()), returned);
  EXPECT_EQ(handle_on(stream), returned);
}

// A hipBLASLt handle owns the synchronizer its split-K kernels spin on.
TEST(CUDABlasHandlePoolTest, LtHandleFromExitedThreadStaysOnItsStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  expectHandleFromExitedThreadStaysOnItsStream(
      [] { return at::cuda::getCurrentCUDABlasLtHandle(); },
      c10::cuda::getStreamFromPool());
}

// The same for a stream c10 does not own. The pool never touches the stream,
// so its owner may destroy it at any time.
TEST(CUDABlasHandlePoolTest, LtHandleFromExitedThreadStaysOnItsExternalStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  cudaStream_t external = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&external, cudaStreamNonBlocking), cudaSuccess);
  auto destroy_external =
      c10::make_scope_exit([&] { (void)cudaStreamDestroy(external); });
  expectHandleFromExitedThreadStaysOnItsStream(
      [] { return at::cuda::getCurrentCUDABlasLtHandle(); },
      c10::cuda::getStreamFromExternal(external, 0));
}

// A public rocBLAS handle keeps the workspace bound at its creation.
TEST(CUDABlasHandlePoolTest, PublicHandleFromExitedThreadStaysOnItsStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (at::cuda::isCUDABlasWorkspaceCachingEnabled()) {
    GTEST_SKIP() << "requires eager workspaces";
  }
  expectHandleFromExitedThreadStaysOnItsStream(
      [] { return at::cuda::getCurrentCUDABlasHandle(); },
      c10::cuda::getStreamFromPool());
}

namespace {

// ATen's internal rocBLAS handle for eager work, which embeds a hipBLASLt
// handle where rocBLAS uses its hipBLASLt backend.
cublasHandle_t currentInternalCublasHandle() {
  auto scoped = at::cuda::getCurrentCUDABlasHandleWithWorkspace();
  return scoped;
}

} // namespace

TEST(CUDABlasHandlePoolTest, EagerInternalHandleIsPerStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  const auto handle_on = [](c10::cuda::CUDAStream stream) {
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    return currentInternalCublasHandle();
  };
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  const auto first_handle = handle_on(first);
  EXPECT_NE(handle_on(second), first_handle);
  EXPECT_EQ(handle_on(first), first_handle);
}

TEST(CUDABlasHandlePoolTest, InternalHandleFromExitedThreadStaysOnItsStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  expectHandleFromExitedThreadStaysOnItsStream(
      [] { return currentInternalCublasHandle(); },
      c10::cuda::getStreamFromPool());
}

// Requesting the public handle is how a thread creates its handles before a
// capture, so it also reserves the internal handle that the thread's eager
// operations on the stream use.
TEST(CUDABlasHandlePoolTest, PublicHandleRequestReservesEagerInternalHandle) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  const auto stream = c10::cuda::getStreamFromPool();
  const auto on_stream = [&](auto body) {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    body();
  };
  // Returned last for `stream`, so it is the next internal handle the pool
  // hands out for it.
  cublasHandle_t returned = nullptr;
  runOnThread(
      [&] { on_stream([&] { returned = currentInternalCublasHandle(); }); });
  // A thread that requests only the public handle keeps its handles while
  // another thread takes an internal handle for the stream.
  std::promise<void> requested;
  std::promise<void> release;
  std::exception_ptr failure;
  std::thread requester([&, released = release.get_future()] {
    try {
      on_stream([] { (void)at::cuda::getCurrentCUDABlasHandle(); });
    } catch (...) {
      failure = std::current_exception();
    }
    requested.set_value();
    released.wait();
  });
  auto join_requester = c10::make_scope_exit([&] {
    release.set_value();
    requester.join();
  });
  requested.get_future().wait();
  if (failure) {
    std::rethrow_exception(failure);
  }
  cublasHandle_t next = nullptr;
  runOnThread([&] { on_stream([&] { next = currentInternalCublasHandle(); }); });

  EXPECT_NE(next, returned);
}

// A stream first used mid-capture cannot create a handle, so every stream in a
// capture uses the thread's internal handle rather than a stream's eager one.
TEST(CUDABlasHandlePoolTest, CaptureUsesThreadsInternalHandle) {
  if (!at::cuda::is_available()) {
    return;
  }
  const auto prev_backend = at::globalContext().blasPreferredBackend();
  at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublas);
  auto restore_backend = c10::make_scope_exit(
      [&] { at::globalContext().setBlasPreferredBackend(prev_backend); });
  at::cuda::CUDAGuard device_guard(0);
  const auto capture_stream = c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
  const auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  const auto a = at::randn({64, 64}, options);
  const auto expected = at::mm(a, a);
  const auto eager = currentInternalCublasHandle();
  // A new stream that this thread first uses inside the capture.
  hipStream_t raw_side = nullptr;
  ASSERT_EQ(
      hipStreamCreateWithFlags(&raw_side, hipStreamNonBlocking), hipSuccess);
  auto destroy_side = c10::make_scope_exit([&] {
    (void)hipStreamSynchronize(raw_side);
    (void)hipStreamDestroy(raw_side);
  });
  const auto side = c10::cuda::getStreamFromExternal(raw_side, 0);

  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  cublasHandle_t captured_handle = nullptr;
  cublasHandle_t side_handle = nullptr;
  at::Tensor captured;
  at::Tensor side_captured;
  try {
    captured_handle = currentInternalCublasHandle();
    captured = at::mm(a, a);
    at::cuda::CUDAEvent forked;
    forked.record(capture_stream);
    forked.block(side);
    {
      c10::cuda::CUDAStreamGuard side_guard(side);
      side_handle = currentInternalCublasHandle();
      side_captured = at::mm(a, a);
    }
    at::cuda::CUDAEvent joined;
    joined.record(side);
    joined.block(capture_stream);
  } catch (...) {
    // Ends the capture, so that a failure here leaves no capture running.
    try {
      graph.capture_end();
    } catch (...) {
    }
    throw;
  }
  graph.capture_end();
  graph.replay();
  capture_stream.synchronize();

  EXPECT_NE(captured_handle, eager);
  EXPECT_EQ(side_handle, captured_handle);
  EXPECT_TRUE(at::allclose(captured, expected));
  EXPECT_TRUE(at::allclose(side_captured, expected));
}

// The next thread to take a handle must not wait for an exited thread's work,
// which may itself wait for work that the next thread queues later. Here the
// exited thread's stream waits for a value that only the next thread writes.
TEST(CUDABlasHandlePoolTest, LtHandleTakeDoesNotWaitForExitedThreadsWork) {
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
  // HIP maps streams onto a few hardware queues per priority, and a wait blocks
  // every stream on its queue. So the gated stream `first` has a priority of
  // its own, and so does `opener`, which opens the gate when `second` cannot,
  // as when a regression makes `second` wait for `first`.
  int least_priority = 0;
  int greatest_priority = 0;
  ASSERT_EQ(
      hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
      hipSuccess);
  hipStream_t raw_opener = nullptr;
  ASSERT_EQ(
      hipStreamCreateWithPriority(
          &raw_opener, hipStreamNonBlocking, least_priority),
      hipSuccess);
  auto destroy_opener =
      c10::make_scope_exit([&] { (void)hipStreamDestroy(raw_opener); });
  const auto opener = c10::cuda::getStreamFromExternal(raw_opener, 0);
  const auto first = c10::cuda::getStreamFromPool(/*isHighPriority=*/true);
  const auto second = c10::cuda::getStreamFromPool();
  // Both threads take spares, so neither creates a handle while the gate is
  // closed.
  at::cuda::ensureCublasLtHandlesAvailable(2);
  void* gate = nullptr;
  ASSERT_EQ(
      hipExtMallocWithFlags(&gate, sizeof(uint64_t), hipMallocSignalMemory),
      hipSuccess);
  // No stream may still be using the gate when it is freed.
  auto free_gate = c10::make_scope_exit([&] {
    (void)hipStreamWriteValue64(opener, gate, 1, 0);
    (void)hipStreamSynchronize(first);
    (void)hipStreamSynchronize(second);
    (void)hipFree(gate);
  });
  ASSERT_EQ(hipStreamWriteValue64(first, gate, 0, 0), hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(first), hipSuccess);
  // Taking a handle and returning it must not wait for the gated work. Both
  // are waited for against one deadline, so a regression that waits fails the
  // test instead of hanging it.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  cublasLtHandle_t returned = nullptr;
  std::promise<void> owner_exited;
  auto owner_exit = owner_exited.get_future();
  std::exception_ptr owner_failure;
  std::thread owner([&] {
    // Ready once the thread has returned its handles.
    owner_exited.set_value_at_thread_exit();
    try {
      at::cuda::CUDAGuard guard(0);
      c10::cuda::CUDAStreamGuard stream_guard(first);
      returned = at::cuda::getCurrentCUDABlasLtHandle();
      EXPECT_EQ(
          hipStreamWaitValue64(first, gate, 1, hipStreamWaitValueGte),
          hipSuccess);
    } catch (...) {
      owner_failure = std::current_exception();
    }
  });
  const bool owner_exited_in_time =
      owner_exit.wait_until(deadline) == std::future_status::ready;

  cublasLtHandle_t reused = nullptr;
  at::cuda::CUDAEvent opened;
  std::atomic<bool> launched{false};
  std::exception_ptr next_owner_failure;
  std::thread next_owner;
  if (owner_exited_in_time) {
    next_owner = std::thread([&] {
      try {
        at::cuda::CUDAGuard guard(0);
        c10::cuda::CUDAStreamGuard stream_guard(second);
        reused = at::cuda::getCurrentCUDABlasLtHandle();
        EXPECT_EQ(hipStreamWriteValue64(second, gate, 1, 0), hipSuccess);
        opened.record(second);
        launched = true;
      } catch (...) {
        next_owner_failure = std::current_exception();
      }
    });
  }
  // Does not throw, so the threads are always joined below.
  const auto gate_opened = [&] {
    return launched && hipEventQuery(opened.event()) == hipSuccess;
  };
  while (next_owner.joinable() && !gate_opened() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool opened_in_time = gate_opened();
  // Lets any thread or stream still waiting on the gate finish.
  EXPECT_EQ(hipStreamWriteValue64(opener, gate, 1, 0), hipSuccess);
  owner.join();
  if (next_owner.joinable()) {
    next_owner.join();
  }
  for (const auto& failure : {owner_failure, next_owner_failure}) {
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

  EXPECT_TRUE(owner_exited_in_time);
  EXPECT_TRUE(opened_in_time);
  EXPECT_NE(reused, returned);
}

// A stream's list of returned handles outlives the stream and is reused by a
// new stream created at the same address. That is safe only because
// hipStreamDestroy waits for the stream's queued work, which HIP does not
// promise, so this checks it.
TEST(CUDABlasHandlePoolTest, StreamDestroyWaitsForQueuedWork) {
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
  void* ran = nullptr;
  hipStream_t stream = nullptr;
  std::thread destroyer;
  bool destroy_hung = false;
  const auto opener = c10::cuda::getStreamFromPool();
  // Runs on every exit, so no stream is left waiting on the gate. If the gate
  // cannot be opened or the destroy never returned, the device's state is
  // unknown, so the destroying thread is detached and the memory is leaked.
  auto free_memory = c10::make_scope_exit([&] {
    const bool opened = gate == nullptr ||
        (hipStreamWriteValue64(opener, gate, 1, 0) == hipSuccess &&
         hipStreamSynchronize(opener) == hipSuccess);
    if (destroy_hung || !opened) {
      if (destroyer.joinable()) {
        destroyer.detach();
      }
      return;
    }
    if (destroyer.joinable()) {
      destroyer.join();
    } else if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
    (void)hipDeviceSynchronize();
    (void)hipFree(gate);
    (void)hipFree(ran);
  });
  ASSERT_EQ(
      hipExtMallocWithFlags(&gate, sizeof(uint64_t), hipMallocSignalMemory),
      hipSuccess);
  ASSERT_EQ(
      hipExtMallocWithFlags(&ran, sizeof(uint64_t), hipMallocSignalMemory),
      hipSuccess);
  ASSERT_EQ(hipStreamWriteValue64(opener, gate, 0, 0), hipSuccess);
  ASSERT_EQ(hipStreamWriteValue64(opener, ran, 0, 0), hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(opener), hipSuccess);
  // A priority that `opener` does not have, so the two never share a hardware
  // queue, which the wait would block.
  int least_priority = 0;
  int greatest_priority = 0;
  ASSERT_EQ(
      hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
      hipSuccess);
  ASSERT_EQ(
      hipStreamCreateWithPriority(
          &stream, hipStreamNonBlocking, greatest_priority),
      hipSuccess);
  ASSERT_EQ(
      hipStreamWaitValue64(stream, gate, 1, hipStreamWaitValueGte), hipSuccess);
  ASSERT_EQ(hipStreamWriteValue64(stream, ran, 1, 0), hipSuccess);

  // Shared with the destroying thread, which may outlive the test if
  // hipStreamDestroy never returns.
  struct DestroyResult {
    std::atomic<bool> done{false};
    std::atomic<hipError_t> status{hipErrorUnknown};
  };
  const auto result = std::make_shared<DestroyResult>();
  destroyer = std::thread([result, stream] {
    result->status.store(hipStreamDestroy(stream));
    result->done = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  const bool destroyed_before_gate_opened = result->done;
  ASSERT_EQ(hipStreamWriteValue64(opener, gate, 1, 0), hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(opener), hipSuccess);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!result->done && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!result->done) {
    destroy_hung = true;
    destroyer.detach();
    FAIL() << "hipStreamDestroy did not return within 10 s of the gate opening";
  }

  EXPECT_FALSE(destroyed_before_gate_opened);
  EXPECT_EQ(result->status, hipSuccess);
  uint64_t ran_value = 0;
  ASSERT_EQ(
      hipMemcpy(&ran_value, ran, sizeof(ran_value), hipMemcpyDeviceToHost),
      hipSuccess);
  EXPECT_EQ(ran_value, 1u);
}

namespace {

// Run in a fresh process by LtHandleCreationWaitsForLegacyStream.
void expectLtHandleCreationWaitsForLegacyStream() {
  at::cuda::CUDAGuard device_guard(0);
  // Creates this thread's handle first, so that the creation timed below does
  // not include loading hipBLASLt.
  (void)at::cuda::getCurrentCUDABlasLtHandle();
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  hipStream_t fresh = nullptr;
  ASSERT_EQ(hipStreamCreateWithFlags(&fresh, hipStreamNonBlocking), hipSuccess);
  // The legacy stream has the default priority, so an opener with the greatest
  // one never shares its hardware queue, which the wait would block.
  int least_priority = 0;
  int greatest_priority = 0;
  ASSERT_EQ(
      hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
      hipSuccess);
  hipStream_t opener = nullptr;
  ASSERT_EQ(
      hipStreamCreateWithPriority(
          &opener, hipStreamNonBlocking, greatest_priority),
      hipSuccess);
  void* gate = nullptr;
  ASSERT_EQ(
      hipExtMallocWithFlags(&gate, sizeof(uint64_t), hipMallocSignalMemory),
      hipSuccess);
  ASSERT_EQ(hipStreamWriteValue64(opener, gate, 0, 0), hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(opener), hipSuccess);
  ASSERT_EQ(
      hipStreamWaitValue64(nullptr, gate, 1, hipStreamWaitValueGte),
      hipSuccess);
  // Runs on every exit, so no failure leaves the legacy stream waiting.
  auto open_gate = c10::make_scope_exit(
      [&] { (void)hipStreamWriteValue64(opener, gate, 1, 0); });

  // Shared with the creating thread, which may outlive this function if the
  // creation never returns.
  const auto created = std::make_shared<std::atomic<bool>>(false);
  std::thread creator([created, fresh] {
    c10::cuda::CUDAStreamGuard stream_guard(
        c10::cuda::getStreamFromExternal(fresh, 0));
    (void)at::cuda::getCurrentCUDABlasLtHandle();
    *created = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  const bool created_while_held = *created;
  ASSERT_EQ(hipStreamWriteValue64(opener, gate, 1, 0), hipSuccess);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!*created && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // The parent of a death test shows this process's stderr, not its failures.
  const auto fail = [](const char* why) {
    std::cerr << why << std::endl;
    ADD_FAILURE() << why;
  };
  if (!*created) {
    creator.detach();
    fail("Creating the handle did not return within 10 s of the gate opening");
    return;
  }
  creator.join();
  if (created_while_held) {
    fail("Creating the handle returned while the legacy stream was held");
  }
}

} // namespace

// hipblasLtCreate zeroes a new handle's synchronizer with a hipMemset still
// queued on the legacy stream (see Note [hipblaslt handles]), so creating a
// handle must not return while the legacy stream is held.
TEST(CUDABlasHandlePoolDeathTest, LtHandleCreationWaitsForLegacyStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  int can_wait = 0;
  ASSERT_EQ(
      hipDeviceGetAttribute(
          &can_wait, hipDeviceAttributeCanUseStreamWaitValue, 0),
      hipSuccess);
  if (can_wait == 0) {
    GTEST_SKIP() << "requires hipStreamWaitValue64";
  }
  // Re-exec for a fresh pool, so that the handle is created rather than taken
  // from the spares other tests stocked.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        expectLtHandleCreationWaitsForLegacyStream();
        // Propagate child assertion failures to the parent.
        std::_Exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS),
      "");
}

// A thread that first needs a handle mid-capture, like an autograd worker in
// backward, cannot create one, so CUDAGraph::capture_begin() stocks a spare.
// The handles exited threads returned belong to their streams and must not
// count as that spare.
TEST(CUDABlasHandlePoolTest, LtHandleUnderCaptureTakesUnusedSpare) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAGuard device_guard(0);
  const auto first = c10::cuda::getStreamFromPool();
  const auto second = c10::cuda::getStreamFromPool();
  // Threads that hold every handle a thread on `second` can get without
  // creating one: those that earlier tests in this process returned for it,
  // then the spares. Afterwards the pool's lists hold only the handle the
  // exited thread below returns and the spare stocked after it.
  std::promise<void> release_holders;
  const auto holders_released = release_holders.get_future().share();
  std::vector<std::thread> holders;
  auto join_holders = c10::make_scope_exit([&] {
    release_holders.set_value();
    for (auto& holder : holders) {
      holder.join();
    }
  });
  // Bounded, in case a lookup that must not create a handle does.
  constexpr size_t kMaxHolders = 64;
  bool drained = false;
  while (!drained && holders.size() < kMaxHolders) {
    std::promise<cublasLtHandle_t> held;
    auto held_future = held.get_future();
    holders.emplace_back([&, held = std::move(held)]() mutable {
      try {
        at::cuda::CUDAGuard guard(0);
        c10::cuda::CUDAStreamGuard stream_guard(second);
        held.set_value(at::cuda::getCurrentCUDABlasLtHandleIfAvailable());
      } catch (...) {
        held.set_exception(std::current_exception());
      }
      holders_released.wait();
    });
    drained = held_future.get() == nullptr;
  }
  ASSERT_TRUE(drained);
  cublasLtHandle_t returned = nullptr;
  runOnThread([&] {
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(first);
    returned = at::cuda::getCurrentCUDABlasLtHandle();
  });

  at::cuda::ensureCublasLtHandlesAvailable(1);
  ASSERT_EQ(
      cudaStreamBeginCapture(second, cudaStreamCaptureModeGlobal), cudaSuccess);
  cublasLtHandle_t reused = nullptr;
  std::exception_ptr failure;
  std::thread worker([&] {
    try {
      at::cuda::CUDAGuard guard(0);
      c10::cuda::CUDAStreamGuard stream_guard(second);
      reused = at::cuda::getCurrentCUDABlasLtHandle();
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
  EXPECT_NE(reused, nullptr);
  EXPECT_NE(reused, returned);
}

namespace {
// Counts the warnings about hipStreamPerThread that this thread emits.
struct PerThreadStreamWarnings : c10::WarningHandler {
  int count = 0;
  void process(const c10::Warning& warning) override {
    count += warning.msg().find("hipStreamPerThread") != std::string::npos;
  }
};
} // namespace

// The warning about a kept hipStreamPerThread handle prints exactly once for
// all pools, and not for a lookup that keeps no handle.
TEST(CUDABlasHandlePoolTest, PerThreadStreamWarnsOnceWhenHandleKept) {
  if (!at::cuda::is_available()) {
    return;
  }
  if (kept_per_thread_stream_handle) {
    GTEST_SKIP() << "an earlier test in this process kept a hipStreamPerThread "
                    "handle, so the warning has already printed";
  }
  kept_per_thread_stream_handle = true;
  cublasLtHandle_t looked_up = nullptr;
  int after_lookup = 0;
  int total = 0;
  runOnThread([&] {
    PerThreadStreamWarnings warnings;
    c10::WarningUtils::WarningHandlerGuard handler_guard(&warnings);
    at::cuda::CUDAGuard guard(0);
    c10::cuda::CUDAStreamGuard stream_guard(
        c10::cuda::getStreamFromExternal(hipStreamPerThread, 0));
    looked_up = at::cuda::getCurrentCUDABlasLtHandleIfAvailable();
    after_lookup = warnings.count;
    (void)at::cuda::getCurrentCUDABlasLtHandle();
    (void)at::cuda::getCurrentCUDABlasHandle();
    (void)currentInternalCublasHandle();
    total = warnings.count;
  });
  if (looked_up == nullptr) {
    EXPECT_EQ(after_lookup, 0);
  }
  EXPECT_EQ(total, 1);
}

// hipStreamPerThread is a different stream in each thread, so a handle used on
// it is not returned to the pool.
TEST(CUDABlasHandlePoolTest, LtHandleForPerThreadStreamIsNotReused) {
  if (!at::cuda::is_available()) {
    return;
  }
  kept_per_thread_stream_handle = true;
  const auto handle_on_per_thread_stream = [] {
    cublasLtHandle_t handle = nullptr;
    runOnThread([&] {
      at::cuda::CUDAGuard guard(0);
      c10::cuda::CUDAStreamGuard stream_guard(
          c10::cuda::getStreamFromExternal(hipStreamPerThread, 0));
      handle = at::cuda::getCurrentCUDABlasLtHandle();
    });
    return handle;
  };
  const auto first = handle_on_per_thread_stream();
  EXPECT_NE(handle_on_per_thread_stream(), first);
}

// The same on the fake pool, without a GPU. Defined after
// PerThreadStreamWarnsOnceWhenHandleKept so that, in the default order, that
// test runs before this one prints the warning.
TEST(CUDABlasHandlePoolTest, PerThreadStreamHandleIsNotReturned) {
  kept_per_thread_stream_handle = true;
  auto pool = std::make_shared<FakeHandlePool>();
  void* const per_thread_stream = static_cast<void*>(hipStreamPerThread);
  int* const kept = reserveAndExit(*pool, 0, per_thread_stream);
  EXPECT_NE(reserveAndExit(*pool, 0, per_thread_stream), kept);
  EXPECT_NE(reserveAndExit(*pool, 0, reinterpret_cast<void*>(0x10)), kept);
}

#endif // USE_ROCM

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
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
