#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

// Benchmark for cuBLAS/cuBLASLt workspace acquisition overhead.
//
// Sequential mode: spawns one thread at a time, each calling the workspace API
// `calls_per_thread` times, then joining before spawning the next. With
// calls_per_thread=1 this is the adversarial cold-path case for thread_local
// workspaces: every thread must allocate fresh.
//
// Concurrent mode: spawns two threads at a time to add light contention on the
// shared_mutex path (main) vs independent thread_local paths (optimization).

namespace {

void bench_cublas(int calls_per_thread, int num_threads, int concurrency) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; i += concurrency) {
    std::vector<std::thread> threads;
    int batch = std::min(concurrency, num_threads - i);
    for (int t = 0; t < batch; ++t) {
      threads.emplace_back([&]() {
        at::cuda::CUDAGuard guard(0);
        for (int j = 0; j < calls_per_thread; ++j) {
          auto handle = at::cuda::getCurrentCUDABlasHandle();
          (void)handle;
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  int total_calls = num_threads * calls_per_thread;
  printf(
      "[cuBLAS]   threads=%5d  calls/thread=%4d  total=%.2f ms  "
      "per-thread=%.4f ms  per-iter=%.4f us\n",
      num_threads,
      calls_per_thread,
      elapsed_ms,
      elapsed_ms / num_threads,
      elapsed_ms * 1000.0 / total_calls);
}

void bench_cublaslt(int calls_per_thread, int num_threads, int concurrency) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; i += concurrency) {
    std::vector<std::thread> threads;
    int batch = std::min(concurrency, num_threads - i);
    for (int t = 0; t < batch; ++t) {
      threads.emplace_back([&]() {
        at::cuda::CUDAGuard guard(0);
        // Ensure CUDA runtime TLS is initialized (in real usage the context
        // is always primed before workspace acquisition via prior CUDA calls)
        cudaFree(nullptr);
        for (int j = 0; j < calls_per_thread; ++j) {
          auto* ws = at::cuda::getCUDABlasLtWorkspace();
          (void)ws;
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  int total_calls = num_threads * calls_per_thread;
  printf(
      "[cuBLASLt] threads=%5d  calls/thread=%4d  total=%.2f ms  "
      "per-thread=%.4f ms  per-iter=%.4f us\n",
      num_threads,
      calls_per_thread,
      elapsed_ms,
      elapsed_ms / num_threads,
      elapsed_ms * 1000.0 / total_calls);
}

struct Config {
  int calls_per_thread;
  int num_threads;
};

Config configs[] = {
    {1, 20000},
    {10, 10000},
    {100, 5000},
    {1000, 2000},
};

} // namespace

TEST(CUDABlasWorkspaceBench, Sequential) {
  if (!at::cuda::is_available()) {
    return;
  }

  // Warmup: prime CUDA context
  {
    std::thread t([]() {
      at::cuda::CUDAGuard guard(0);
      auto handle = at::cuda::getCurrentCUDABlasHandle();
      auto* ws = at::cuda::getCUDABlasLtWorkspace();
      (void)handle;
      (void)ws;
    });
    t.join();
  }

  printf("Sequential (1 thread at a time):\n");
  for (auto& cfg : configs) {
    bench_cublas(cfg.calls_per_thread, cfg.num_threads, 1);
    bench_cublaslt(cfg.calls_per_thread, cfg.num_threads, 1);
    printf("\n");
  }
}

TEST(CUDABlasWorkspaceBench, Concurrent2) {
  if (!at::cuda::is_available()) {
    return;
  }

  printf("Concurrent (2 threads at a time):\n");
  for (auto& cfg : configs) {
    bench_cublas(cfg.calls_per_thread, cfg.num_threads, 2);
    bench_cublaslt(cfg.calls_per_thread, cfg.num_threads, 2);
    printf("\n");
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(1);
  return RUN_ALL_TESTS();
}
