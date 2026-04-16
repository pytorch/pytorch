#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <atomic>
#include <thread>
#include <vector>

// Test that concurrent threads can independently use cuBLAS handles
// and workspaces without interference.
// With thread_local workspaces, each thread has its own workspace map
// so there is no shared state to race on.

TEST(CUDABlasHandlePoolTest, ConcurrentHandleAndWorkspaceAccess) {
  if (!at::cuda::is_available()) {
    return;
  }

  constexpr int num_threads = 20;
  constexpr int iterations_per_thread = 50;

  std::atomic<int> error_count{0};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&error_count]() {
      try {
        at::cuda::CUDAGuard device_guard(0);

        for (int j = 0; j < iterations_per_thread; ++j) {
          const auto handle = at::cuda::getCurrentCUDABlasHandle();
          const auto workspace = at::cuda::getCUDABlasLtWorkspace();

          if (handle == nullptr || workspace == nullptr) {
            error_count++;
          }

          // Clearing workspaces on this thread should not affect other threads
          if (j % 10 == 0) {
            at::cuda::clearCublasWorkspaces();
          }
        }
      } catch (const std::exception&) {
        error_count++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(1);
  return RUN_ALL_TESTS();
}
