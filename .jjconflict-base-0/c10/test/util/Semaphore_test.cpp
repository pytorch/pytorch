#include <c10/util/Semaphore.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>

#include <thread>

using namespace ::testing;

TEST(SemaphoreTest, TestConcurrency) {
  auto num_threads = std::thread::hardware_concurrency();
  auto num_incr = 10000;

  c10::Semaphore sem;

  std::vector<std::thread> threads;
  for ([[maybe_unused]] const auto _ : c10::irange(num_threads)) {
    threads.emplace_back([num_incr = num_incr, &sem]() {
      for ([[maybe_unused]] const auto _ : c10::irange(num_incr)) {
        sem.release();
      }
      for ([[maybe_unused]] const auto _ : c10::irange(num_incr)) {
        sem.acquire();
      }
      sem.release(num_incr);
      for ([[maybe_unused]] const auto _ : c10::irange(num_incr)) {
        sem.acquire();
      }
    });
  }

  std::for_each(
      threads.begin(), threads.end(), [](std::thread& t) { t.join(); });

  EXPECT_FALSE(sem.tryAcquire());
}
