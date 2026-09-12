#include <c10/core/CPUAllocator.h>
#include <c10/mobile/CPUCachingAllocator.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <thread>
#include <vector>

namespace {

TEST(CPUCachingAllocatorTest, ConcurrentDestructionPreservesLiveAllocations) {
  std::atomic<bool> start{false};
  std::array<std::thread, 4> threads;
  for (auto& thread : threads) {
    thread = std::thread([&start]() {
      auto* allocator = c10::GetDefaultMobileCPUAllocator();
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < 1000; ++iteration) {
        c10::DataPtr live;
        {
          c10::CPUCachingAllocator caching_allocator;
          c10::WithCPUCachingAllocatorGuard guard(&caching_allocator);
          live = allocator->allocate(sizeof(int));
          *static_cast<int*>(live.get()) = iteration;
          std::vector<c10::DataPtr> cached;
          for (int block = 1; block <= 16; ++block) {
            cached.emplace_back(allocator->allocate(64 * block));
          }
        }
        EXPECT_EQ(iteration, *static_cast<int*>(live.get()));
      }
    });
  }
  start.store(true);
  for (auto& thread : threads) {
    thread.join();
  }
}

} // namespace
